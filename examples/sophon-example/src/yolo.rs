use anyhow::Context;
use image::{DynamicImage, GenericImage, Rgb, RgbImage};
use serde::Serialize;
use sophon_runtime::DType;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct Letterbox {
    pub tx: f32,
    pub ty: f32,
    pub ratio: f32,
    pub src_w: u32,
    pub src_h: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Det {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class_id: i32,
}

#[derive(Clone, Debug)]
pub struct Preprocessed {
    pub name: String,
    pub orig: RgbImage,
    pub input_bytes: Vec<u8>,
    pub letterbox: Letterbox,
}

#[derive(Serialize)]
pub struct ImageResult {
    pub image_name: String,
    pub bboxes: Vec<BBoxJson>,
}

#[derive(Serialize)]
pub struct BBoxJson {
    pub category_id: i32,
    pub score: f32,
    pub bbox: [f32; 4],
}

pub fn read_class_names(path: &Path) -> anyhow::Result<Vec<String>> {
    let s = std::fs::read_to_string(path).with_context(|| format!("读取 classnames 失败: {}", path.display()))?;
    let mut out = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.to_string());
    }
    Ok(out)
}

pub fn list_images(input: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let meta = std::fs::metadata(input).with_context(|| format!("读取输入路径失败: {}", input.display()))?;
    if meta.is_dir() {
        let mut v = Vec::new();
        for ent in std::fs::read_dir(input).with_context(|| format!("遍历目录失败: {}", input.display()))? {
            let ent = ent?;
            let p = ent.path();
            if !p.is_file() {
                continue;
            }
            let ext = p
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" {
                v.push(p);
            }
        }
        v.sort();
        Ok(v)
    } else {
        Ok(vec![input.to_path_buf()])
    }
}

pub fn preprocess(
    img: DynamicImage,
    name: String,
    net_w: u32,
    net_h: u32,
    input_dtype: DType,
    input_scale: f32,
) -> anyhow::Result<Preprocessed> {
    let orig = img.to_rgb8();
    let (src_w, src_h) = orig.dimensions();
    let (ratio, is_align_width) = aspect_ratio(src_w, src_h, net_w, net_h);
    let new_w = if is_align_width { net_w } else { ((src_w as f32) * ratio).round().max(1.0) as u32 };
    let new_h = if is_align_width { ((src_h as f32) * ratio).round().max(1.0) as u32 } else { net_h };
    let tx = ((net_w as i32 - new_w as i32) / 2).max(0) as u32;
    let ty = ((net_h as i32 - new_h as i32) / 2).max(0) as u32;

    let resized = image::imageops::resize(&orig, new_w, new_h, image::imageops::FilterType::Triangle);
    let mut canvas = RgbImage::from_pixel(net_w, net_h, Rgb([114, 114, 114]));
    canvas.copy_from(&resized, tx, ty)?;

    let alpha = input_scale / 255.0;
    let input_bytes = match input_dtype {
        DType::F32 => rgb_planar_f32_bytes(&canvas, alpha),
        DType::I8 => rgb_planar_i8_bytes(&canvas, alpha),
        DType::U8 => rgb_planar_u8_bytes(&canvas, alpha),
        DType::F16 | DType::I16 | DType::U16 | DType::I32 | DType::U32 => {
            anyhow::bail!("暂不支持的输入 dtype: {:?}", input_dtype)
        }
    };

    Ok(Preprocessed {
        name,
        orig,
        input_bytes,
        letterbox: Letterbox {
            tx: tx as f32,
            ty: ty as f32,
            ratio,
            src_w,
            src_h,
        },
    })
}

pub fn decode_yolov8_like(
    output: &[f32],
    output_shape: &[i32],
    conf_thresh: f32,
    nms_thresh: f32,
    class_num: usize,
    max_det: usize,
    agnostic: bool,
    letterbox: Letterbox,
) -> Vec<Det> {
    let (bs, box_num, nout, is_transposed) = match output_shape {
        [bs, a, b] => {
            if a >= b {
                (*bs as usize, *a as usize, *b as usize, true)
            } else {
                (*bs as usize, *b as usize, *a as usize, false)
            }
        }
        _ => return Vec::new(),
    };
    if bs == 0 || box_num == 0 || nout < 4 {
        return Vec::new();
    }

    let offset = if is_transposed { 1 } else { box_num };
    let batch_stride = box_num * nout;

    let mut all = Vec::new();
    for bi in 0..bs {
        let base = bi * batch_stride;
        let mut cand = Vec::new();
        for i in 0..box_num {
            let box_index = if is_transposed { i * nout } else { i };
            let center_x = output[base + box_index];
            let center_y = output[base + box_index + 1 * offset];
            let w = output[base + box_index + 2 * offset];
            let h = output[base + box_index + 3 * offset];

            for cid in 0..class_num {
                let score = output[base + box_index + (4 + cid) * offset];
                if score <= conf_thresh {
                    continue;
                }
                let c = if agnostic { 0.0 } else { (cid as f32) * 7680.0 };
                let mut x1 = center_x - w / 2.0 + c;
                let mut y1 = center_y - h / 2.0 + c;
                let mut x2 = x1 + w;
                let mut y2 = y1 + h;
                x1 = (x1 - letterbox.tx) / letterbox.ratio;
                y1 = (y1 - letterbox.ty) / letterbox.ratio;
                x2 = (x2 - letterbox.tx) / letterbox.ratio;
                y2 = (y2 - letterbox.ty) / letterbox.ratio;
                cand.push(Det {
                    x1,
                    y1,
                    x2,
                    y2,
                    score,
                    class_id: cid as i32,
                });
            }
        }
        let mut dets = nms(cand, nms_thresh);
        if dets.len() > max_det {
            dets.drain(0..(dets.len() - max_det));
        }
        if !agnostic {
            for d in dets.iter_mut() {
                let c = (d.class_id as f32) * 7680.0;
                d.x1 -= c;
                d.y1 -= c;
                d.x2 -= c;
                d.y2 -= c;
            }
        }
        clip_boxes(&mut dets, letterbox.src_w as f32, letterbox.src_h as f32);
        all.extend(dets);
    }
    all
}

pub fn draw_boxes(img: &mut RgbImage, dets: &[Det], conf_thresh: f32, classnames: Option<&[String]>) {
    for d in dets {
        if d.score < conf_thresh {
            continue;
        }
        let color = palette_color(d.class_id as usize);
        let x1 = d.x1.floor().max(0.0) as i32;
        let y1 = d.y1.floor().max(0.0) as i32;
        let x2 = d.x2.ceil().min(img.width() as f32) as i32;
        let y2 = d.y2.ceil().min(img.height() as f32) as i32;
        draw_rect(img, x1, y1, x2, y2, color, 2);
        let label = if let Some(names) = classnames {
            let name = names.get(d.class_id as usize).map(|s| s.as_str()).unwrap_or("UNK");
            format!("{} {:.2}", name.to_uppercase(), d.score)
        } else {
            format!("ID {} {:.2}", d.class_id, d.score)
        };
        let tw = text_width(&label);
        let th = FONT_H as i32;
        let bx1 = x1;
        let by1 = (y1 - th - 4).max(0);
        let bx2 = (x1 + tw as i32 + 6).min(img.width() as i32);
        let by2 = y1.max(0);
        fill_rect(img, bx1, by1, bx2, by2, [0, 0, 0]);
        draw_text(img, bx1 + 3, by1 + 2, &label, [255, 255, 255]);
    }
}

pub fn to_json_results(dets: &[Det], name: &str) -> ImageResult {
    let mut bboxes = Vec::new();
    for d in dets {
        let w = (d.x2 - d.x1).max(0.0);
        let h = (d.y2 - d.y1).max(0.0);
        bboxes.push(BBoxJson {
            category_id: d.class_id,
            score: d.score,
            bbox: [d.x1, d.y1, w, h],
        });
    }
    ImageResult {
        image_name: name.to_string(),
        bboxes,
    }
}

fn aspect_ratio(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> (f32, bool) {
    let r_w = (dst_w as f32) / (src_w as f32);
    let r_h = (dst_h as f32) / (src_h as f32);
    if r_h > r_w {
        (r_w, true)
    } else {
        (r_h, false)
    }
}

fn rgb_planar_f32_bytes(img: &RgbImage, alpha: f32) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let hw = (w * h) as usize;
    let mut out = Vec::with_capacity(hw * 3 * 4);
    let mut r = vec![0f32; hw];
    let mut g = vec![0f32; hw];
    let mut b = vec![0f32; hw];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y).0;
            let idx = (y * w + x) as usize;
            r[idx] = (p[0] as f32) * alpha;
            g[idx] = (p[1] as f32) * alpha;
            b[idx] = (p[2] as f32) * alpha;
        }
    }
    for v in r.into_iter().chain(g).chain(b) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn rgb_planar_i8_bytes(img: &RgbImage, alpha: f32) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let hw = (w * h) as usize;
    let mut out = vec![0u8; hw * 3];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y).0;
            let idx = (y * w + x) as usize;
            out[idx] = clamp_i8((p[0] as f32) * alpha) as u8;
            out[hw + idx] = clamp_i8((p[1] as f32) * alpha) as u8;
            out[2 * hw + idx] = clamp_i8((p[2] as f32) * alpha) as u8;
        }
    }
    out
}

fn rgb_planar_u8_bytes(img: &RgbImage, alpha: f32) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let hw = (w * h) as usize;
    let mut out = vec![0u8; hw * 3];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y).0;
            let idx = (y * w + x) as usize;
            out[idx] = clamp_u8((p[0] as f32) * alpha);
            out[hw + idx] = clamp_u8((p[1] as f32) * alpha);
            out[2 * hw + idx] = clamp_u8((p[2] as f32) * alpha);
        }
    }
    out
}

fn clamp_i8(v: f32) -> i8 {
    let x = v.round() as i32;
    let x = x.max(-128).min(127);
    x as i8
}

fn clamp_u8(v: f32) -> u8 {
    let x = v.round() as i32;
    let x = x.max(0).min(255);
    x as u8
}

fn clip_boxes(dets: &mut [Det], w: f32, h: f32) {
    for d in dets {
        d.x1 = d.x1.max(0.0).min(w);
        d.y1 = d.y1.max(0.0).min(h);
        d.x2 = d.x2.max(0.0).min(w);
        d.y2 = d.y2.max(0.0).min(h);
    }
}

fn iou(a: &Det, b: &Det) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1).max(0.0) * (a.y2 - a.y1).max(0.0);
    let area_b = (b.x2 - b.x1).max(0.0) * (b.y2 - b.y1).max(0.0);
    inter / (area_a + area_b - inter + 1e-9)
}

fn nms(mut dets: Vec<Det>, thresh: f32) -> Vec<Det> {
    dets.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut index: i32 = dets.len() as i32 - 1;
    while index > 0 {
        let mut i: i32 = 0;
        while i < index {
            let a = dets[index as usize].clone();
            let b = dets[i as usize].clone();
            if iou(&a, &b) > thresh {
                dets.remove(i as usize);
                index -= 1;
            } else {
                i += 1;
            }
        }
        index -= 1;
    }
    dets
}

fn draw_rect(img: &mut RgbImage, x1: i32, y1: i32, x2: i32, y2: i32, color: [u8; 3], thickness: i32) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let x1 = x1.max(0).min(w - 1);
    let x2 = x2.max(0).min(w);
    let y1 = y1.max(0).min(h - 1);
    let y2 = y2.max(0).min(h);
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    for t in 0..thickness {
        let yt = (y1 + t).max(0).min(h - 1);
        let yb = (y2 - 1 - t).max(0).min(h - 1);
        for x in x1..x2 {
            img.put_pixel(x as u32, yt as u32, Rgb(color));
            img.put_pixel(x as u32, yb as u32, Rgb(color));
        }
        let xl = (x1 + t).max(0).min(w - 1);
        let xr = (x2 - 1 - t).max(0).min(w - 1);
        for y in y1..y2 {
            img.put_pixel(xl as u32, y as u32, Rgb(color));
            img.put_pixel(xr as u32, y as u32, Rgb(color));
        }
    }
}

fn fill_rect(img: &mut RgbImage, x1: i32, y1: i32, x2: i32, y2: i32, color: [u8; 3]) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let x1 = x1.max(0).min(w - 1);
    let x2 = x2.max(0).min(w);
    let y1 = y1.max(0).min(h - 1);
    let y2 = y2.max(0).min(h);
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    for y in y1..y2 {
        for x in x1..x2 {
            img.put_pixel(x as u32, y as u32, Rgb(color));
        }
    }
}

const FONT_W: usize = 5;
const FONT_H: usize = 7;

fn text_width(s: &str) -> usize {
    let mut w = 0usize;
    for ch in s.chars() {
        if ch == ' ' {
            w += 3;
        } else if FONT.contains_key(&ch.to_ascii_uppercase()) {
            w += FONT_W + 1;
        } else {
            w += FONT_W + 1;
        }
    }
    w
}

fn draw_text(img: &mut RgbImage, x: i32, y: i32, s: &str, color: [u8; 3]) {
    let mut cx = x;
    for ch in s.chars() {
        if ch == ' ' {
            cx += 3;
            continue;
        }
        let up = ch.to_ascii_uppercase();
        let glyph = FONT.get(&up);
        let bits = if let Some(g) = glyph { g } else { &UNKNOWN_GLYPH };
        for (row_idx, row) in bits.iter().enumerate() {
            for (col_idx, &on) in row.iter().enumerate() {
                if on {
                    let px = cx + col_idx as i32;
                    let py = y + row_idx as i32;
                    if px >= 0 && py >= 0 && px < img.width() as i32 && py < img.height() as i32 {
                        img.put_pixel(px as u32, py as u32, Rgb(color));
                    }
                }
            }
        }
        cx += (FONT_W + 1) as i32;
    }
}

use std::collections::HashMap;
lazy_static::lazy_static! {
    static ref FONT: HashMap<char, [[bool; FONT_W]; FONT_H]> = {
        let mut m: HashMap<char, [[bool; FONT_W]; FONT_H]> = HashMap::new();
        // Digits 0-9
        m.insert('0', [
            [true,true,true,true,true],
            [true,false,false,false,true],
            [true,false,false,true,true],
            [true,false,true,false,true],
            [true,true,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,true],
        ]);
        m.insert('1', [
            [false,false,true,false,false],
            [false,true,true,false,false],
            [true,false,true,false,false],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [true,true,true,true,true],
        ]);
        m.insert('2', [
            [true,true,true,true,true],
            [false,false,false,false,true],
            [false,false,false,false,true],
            [true,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,true,true,true,true],
        ]);
        m.insert('3', [
            [true,true,true,true,true],
            [false,false,false,false,true],
            [false,false,false,false,true],
            [true,true,true,true,true],
            [false,false,false,false,true],
            [false,false,false,false,true],
            [true,true,true,true,true],
        ]);
        m.insert('4', [
            [true,false,false,true,false],
            [true,false,false,true,false],
            [true,false,false,true,false],
            [true,true,true,true,true],
            [false,false,false,true,false],
            [false,false,false,true,false],
            [false,false,false,true,false],
        ]);
        m.insert('5', [
            [true,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,true,true,true,true],
            [false,false,false,false,true],
            [false,false,false,false,true],
            [true,true,true,true,true],
        ]);
        m.insert('6', [
            [true,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,true,true,true,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,true],
        ]);
        m.insert('7', [
            [true,true,true,true,true],
            [false,false,false,false,true],
            [false,false,false,true,false],
            [false,false,true,false,false],
            [false,true,false,false,false],
            [false,true,false,false,false],
            [false,true,false,false,false],
        ]);
        m.insert('8', [
            [true,true,true,true,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,true],
        ]);
        m.insert('9', [
            [true,true,true,true,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,true],
            [false,false,false,false,true],
            [false,false,false,false,true],
            [true,true,true,true,true],
        ]);
        // Letters A-Z (subset adequate for COCO uppercase names)
        m.insert('A', [
            [false,true,true,true,false],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
        ]);
        m.insert('B', [
            [true,true,true,true,false],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,false],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,false],
        ]);
        m.insert('C', [
            [false,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [false,true,true,true,true],
        ]);
        m.insert('D', [
            [true,true,true,true,false],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,false],
        ]);
        m.insert('E', [
            [true,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,true,true,true,true],
        ]);
        m.insert('F', [
            [true,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,false,false,false],
        ]);
        m.insert('G', [
            [false,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,true,true,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [false,true,true,true,true],
        ]);
        m.insert('H', [
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
        ]);
        m.insert('I', [
            [true,true,true,true,true],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [true,true,true,true,true],
        ]);
        m.insert('L', [
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,true,true,true,true],
        ]);
        m.insert('N', [
            [true,false,false,false,true],
            [true,true,false,false,true],
            [true,false,true,false,true],
            [true,false,false,true,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
        ]);
        m.insert('O', *m.get(&'0').unwrap());
        m.insert('P', [
            [true,true,true,true,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,true],
            [true,false,false,false,false],
            [true,false,false,false,false],
            [true,false,false,false,false],
        ]);
        m.insert('R', [
            [true,true,true,true,false],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,false],
            [true,false,true,false,false],
            [true,false,false,true,false],
            [true,false,false,false,true],
        ]);
        m.insert('S', *m.get(&'5').unwrap());
        m.insert('T', [
            [true,true,true,true,true],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [false,false,true,false,false],
            [false,false,true,false,false],
        ]);
        m.insert('U', [
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,false,false,false,true],
            [true,true,true,true,true],
        ]);
        m.insert(':', [
            [false,false,false,false,false],
            [false,false,false,false,false],
            [false,true,false,true,false],
            [false,false,false,false,false],
            [false,true,false,true,false],
            [false,false,false,false,false],
            [false,false,false,false,false],
        ]);
        m.insert('.', [
            [false,false,false,false,false],
            [false,false,false,false,false],
            [false,false,false,false,false],
            [false,false,false,false,false],
            [false,false,false,false,false],
            [false,false,true,false,false],
            [false,false,false,false,false],
        ]);
        m
    };
}

static UNKNOWN_GLYPH: [[bool; FONT_W]; FONT_H] = [
    [true,true,true,true,true],
    [true,false,false,false,true],
    [true,false,true,false,true],
    [true,false,false,false,true],
    [true,false,true,false,true],
    [true,false,false,false,true],
    [true,true,true,true,true],
];
fn palette_color(idx: usize) -> [u8; 3] {
    const COLORS: [[u8; 3]; 25] = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 255, 255],
        [0, 170, 255],
        [0, 85, 255],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
        [255, 0, 0],
        [255, 0, 255],
        [255, 85, 255],
        [255, 170, 255],
        [255, 255, 255],
        [170, 255, 255],
        [85, 255, 255],
    ];
    COLORS[idx % COLORS.len()]
}
