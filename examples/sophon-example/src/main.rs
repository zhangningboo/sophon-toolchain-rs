use clap::Parser;
use sophon_runtime::{DType, Shape, SophonRuntime, Tensor};
use std::path::PathBuf;

mod yolo;

#[derive(Parser, Debug)]
#[command(name = "sophon-example")]
#[command(about = "Rust 版 Sophon bmodel YOLOv8/YOLOv11 目标检测示例")]
struct Args {
    #[arg(long, default_value_t = 0)]
    devid: i32,
    #[arg(long)]
    bmodel: PathBuf,
    #[arg(long)]
    libdir: Option<PathBuf>,
    #[arg(long)]
    input: PathBuf,
    #[arg(long = "conf_thresh", alias = "conf-thresh", default_value_t = 0.25)]
    conf_thresh: f32,
    #[arg(long = "nms_thresh", alias = "nms-thresh", default_value_t = 0.45)]
    nms_thresh: f32,
    #[arg(long)]
    classnames: Option<PathBuf>,
    #[arg(long)]
    net: Option<String>,
    #[arg(long, default_value = "results")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 300)]
    max_det: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if let Some(p) = args.libdir {
        std::env::set_var("SOPHON_SDK_LIBDIR", p);
    }
    let mut rt = SophonRuntime::new_auto(args.devid)?;
    println!("使用后端: {}", rt.backend_name());
    rt.load_bmodel(&args.bmodel)?;
    let nets = rt.networks()?;
    println!("已加载网络: {:?}", nets);

    let net = args.net.clone().unwrap_or_else(|| nets[0].clone());
    let net_info = rt.net_info(&net)?;
    let input0 = net_info
        .inputs
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("网络缺少输入"))?
        .clone();
    let (net_h, net_w, batch) = infer_nchw(&input0.shape)?;

    let class_names = match args.classnames {
        Some(p) => yolo::read_class_names(&p)?,
        None => Vec::new(),
    };

    std::fs::create_dir_all(args.output_dir.join("images"))?;
    let files = yolo::list_images(&args.input)?;
    if files.is_empty() {
        return Err(anyhow::anyhow!("未找到任何图片: {}", args.input.display()));
    }

    let mut results = Vec::new();
    let mut idx = 0usize;
    while idx < files.len() {
        let mut batch_items = Vec::new();
        for _ in 0..batch {
            if idx >= files.len() {
                break;
            }
            let path = &files[idx];
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("image")
                .to_string();
            let img = image::open(path)?;
            let prep = yolo::preprocess(img, name, net_w, net_h, input0.dtype.clone(), input0.scale)?;
            batch_items.push(prep);
            idx += 1;
        }

        let input_tensor = build_batch_input(&input0.dtype, net_h, net_w, &batch_items)?;
        let outs = rt.infer(&net, &[input_tensor])?;

        let (out_idx, out_tensor) = outs
            .iter()
            .enumerate()
            .find(|(_, t)| t.shape.dims.len() == 3)
            .ok_or_else(|| anyhow::anyhow!("未找到 3D 输出张量"))?;
        let out_meta = net_info
            .outputs
            .get(out_idx)
            .cloned()
            .unwrap_or_else(|| sophon_runtime::IoDesc {
                name: format!("output_{}", out_idx),
                dtype: out_tensor.dtype.clone(),
                scale: 1.0,
                shape: out_tensor.shape.clone(),
            });

        let out_f32 = tensor_as_f32(out_tensor, out_meta.scale)?;
        let out_shape: Vec<i32> = out_tensor.shape.dims.clone();
        let class_num = infer_class_num(&out_shape).or_else(|| {
            if !class_names.is_empty() {
                Some(class_names.len())
            } else {
                None
            }
        });
        let class_num = class_num.ok_or_else(|| anyhow::anyhow!("无法推断类别数，请传入 --classnames"))?;

        let (bs, box_num, nout, _) = infer_output_layout(&out_shape)
            .ok_or_else(|| anyhow::anyhow!("输出形状不支持: {:?}", out_shape))?;
        if bs != batch_items.len() {
            return Err(anyhow::anyhow!(
                "batch size 不匹配: 输出 bs={}，输入 batch={}",
                bs,
                batch_items.len()
            ));
        }

        for (bi, item) in batch_items.into_iter().enumerate() {
            let batch_stride = box_num * nout;
            let slice = &out_f32[(bi * batch_stride)..((bi + 1) * batch_stride)];
            let shape_single = [1, out_shape[1], out_shape[2]];
            let dets = yolo::decode_yolov8_like(
                slice,
                &shape_single,
                args.conf_thresh,
                args.nms_thresh,
                class_num,
                args.max_det,
                false,
                item.letterbox,
            );
            let mut vis = item.orig.clone();
            yolo::draw_boxes(&mut vis, &dets, args.conf_thresh, if class_names.is_empty() { None } else { Some(&class_names) });
            let save_path = args.output_dir.join("images").join(&item.name);
            vis.save(save_path)?;
            results.push(yolo::to_json_results(&dets, &item.name));
        }
    }

    let dataset_name = args
        .input
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("input");
    let model_name = args
        .bmodel
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.bmodel");
    let json_file = args
        .output_dir
        .join(format!("{}_{}_rust_result.json", model_name, dataset_name));
    std::fs::write(&json_file, serde_json::to_string_pretty(&results)?)?;
    println!("result saved in {}", json_file.display());
    Ok(())
}

fn infer_nchw(shape: &Shape) -> anyhow::Result<(u32, u32, usize)> {
    if shape.dims.len() != 4 {
        return Err(anyhow::anyhow!("输入形状不是 NCHW: {:?}", shape.dims));
    }
    let n = shape.dims[0].max(1) as usize;
    let h = shape.dims[2].max(1) as u32;
    let w = shape.dims[3].max(1) as u32;
    Ok((h, w, n))
}

fn build_batch_input(dtype: &DType, net_h: u32, net_w: u32, batch: &[yolo::Preprocessed]) -> anyhow::Result<Tensor> {
    let mut data = Vec::new();
    for b in batch {
        data.extend_from_slice(&b.input_bytes);
    }
    let shape = Shape::new(vec![batch.len() as i32, 3, net_h as i32, net_w as i32]);
    Ok(Tensor {
        dtype: dtype.clone(),
        shape,
        data,
    })
}

fn tensor_as_f32(t: &Tensor, scale: f32) -> anyhow::Result<Vec<f32>> {
    match t.dtype {
        DType::F32 => {
            if t.data.len() % 4 != 0 {
                return Err(anyhow::anyhow!("F32 输出字节长度异常: {}", t.data.len()));
            }
            let mut v = Vec::with_capacity(t.data.len() / 4);
            for c in t.data.chunks_exact(4) {
                v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok(v)
        }
        DType::I8 => Ok(t.data.iter().map(|&b| (b as i8) as f32 * scale).collect()),
        DType::U8 => Ok(t.data.iter().map(|&b| (b as f32) * scale).collect()),
        DType::F16 | DType::I16 | DType::U16 | DType::I32 | DType::U32 => {
            Err(anyhow::anyhow!("暂不支持的输出 dtype: {:?}", t.dtype))
        }
    }
}

fn infer_class_num(out_shape: &[i32]) -> Option<usize> {
    match out_shape {
        [_, a, b] => {
            if a >= b {
                Some((*b as usize).saturating_sub(4))
            } else {
                Some((*a as usize).saturating_sub(4))
            }
        }
        _ => None,
    }
}

fn infer_output_layout(out_shape: &[i32]) -> Option<(usize, usize, usize, bool)> {
    match out_shape {
        [bs, a, b] => {
            if *a >= *b {
                Some((*bs as usize, *a as usize, *b as usize, true))
            } else {
                Some((*bs as usize, *b as usize, *a as usize, false))
            }
        }
        _ => None,
    }
}
