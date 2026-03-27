use clap::Parser;
use sophon_runtime::{DType, Shape, SophonRuntime, Tensor};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "sophon-example")]
#[command(about = "Rust 版 Sophon bmodel 加载与推理示例")]
struct Args {
    #[arg(long, default_value_t = 0)]
    devid: i32,
    #[arg(long)]
    bmodel: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut rt = SophonRuntime::new_auto(args.devid)?;
    println!("使用后端: {}", rt.backend_name());
    rt.load_bmodel(&args.bmodel)?;
    let nets = rt.networks()?;
    println!("已加载网络: {:?}", nets);

    let input = Tensor {
        dtype: DType::F32,
        shape: Shape::new(vec![1, 3, 28, 28]),
        data: vec![0u8; 1 * 3 * 28 * 28 * 4],
    };
    let outs = rt.infer(&nets[0], &[input])?;
    println!(
        "输出数量: {}，第一个输出元素数: {}",
        outs.len(),
        outs[0].shape.elements()
    );
    Ok(())
}
