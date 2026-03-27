use std::path::Path;

#[derive(Debug, Clone)]
pub enum DType {
    F32,
    F16,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub dims: Vec<i32>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<i32>>) -> Self {
        Self { dims: dims.into() }
    }
    pub fn elements(&self) -> u64 {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().map(|&d| d as u64).product()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Shape,
    pub data: Vec<u8>,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait RuntimeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn create(devid: i32) -> Result<Box<dyn RuntimeBackend>>
    where
        Self: Sized;
    fn load_bmodel(&mut self, path: &Path) -> Result<()>;
    fn get_networks(&self) -> Result<Vec<String>>;
    fn infer(&self, net_name: &str, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
}

pub struct SophonRuntime {
    backend: Box<dyn RuntimeBackend>,
}

impl SophonRuntime {
    pub fn new_auto(devid: i32) -> Result<Self> {
        #[cfg(feature = "ffi")]
        {
            if let Ok(b) = FfiBackend::create(devid) {
                return Ok(Self { backend: b });
            }
        }
        let b = MockBackend::create(devid)?;
        Ok(Self { backend: b })
    }
    pub fn load_bmodel(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.backend.load_bmodel(path.as_ref())
    }
    pub fn networks(&self) -> Result<Vec<String>> {
        self.backend.get_networks()
    }
    pub fn infer(&self, net_name: &str, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.backend.infer(net_name, inputs)
    }
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }
}

// ---------- Mock backend ----------
struct MockBackend;

impl RuntimeBackend for MockBackend {
    fn name(&self) -> &'static str {
        "mock"
    }
    fn create(_devid: i32) -> Result<Box<dyn RuntimeBackend>>
    where
        Self: Sized,
    {
        Ok(Box::new(MockBackend))
    }
    fn load_bmodel(&mut self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Err(Error::Invalid(format!(
                "bmodel 不存在: {}",
                path.display()
            )));
        }
        Ok(())
    }
    fn get_networks(&self) -> Result<Vec<String>> {
        Ok(vec!["net".to_string()])
    }
    fn infer(&self, _net_name: &str, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(Error::Invalid("输入为空".into()));
        }
        let out_shape = Shape::new(vec![inputs[0].shape.elements() as i32]);
        let out = Tensor {
            dtype: DType::F32,
            shape: out_shape.clone(),
            data: vec![0u8; (out_shape.elements() as usize) * 4],
        };
        Ok(vec![out])
    }
}

// ---------- FFI backend (dynamic load via libloading) ----------
#[cfg(feature = "ffi")]
struct FfiBackend {
    _lib_bmlib: libloading::Library,
    _lib_bmrt: libloading::Library,
}

#[cfg(feature = "ffi")]
impl RuntimeBackend for FfiBackend {
    fn name(&self) -> &'static str {
        "ffi"
    }
    fn create(_devid: i32) -> Result<Box<dyn RuntimeBackend>>
    where
        Self: Sized,
    {
        // 尝试按照常见路径加载动态库，或通过环境变量 SOPHON_SDK_LIBDIR
        let libdir = std::env::var("SOPHON_SDK_LIBDIR").ok();
        let try_paths = |name: &str| -> Vec<std::path::PathBuf> {
            let mut v = Vec::new();
            if let Some(dir) = &libdir {
                v.push(Path::new(dir).join(name));
            }
            // 常见名称（Linux 上 .so，macOS 上 .dylib；此处尽量兼容命名）
            v.push(Path::new(name).to_path_buf());
            v
        };
        let lib_bmlib = {
            let mut last_err: Option<std::io::Error> = None;
            let mut ok = None;
            for p in try_paths("libbmlib.so") // 如需 macOS 测试可改成 .dylib
                .into_iter()
                .chain(try_paths("libbmlib.dylib"))
            {
                match unsafe { libloading::Library::new(&p) } {
                    Ok(l) => {
                        ok = Some(l);
                        break;
                    }
                    Err(e) => last_err = Some(std::io::Error::new(std::io::ErrorKind::NotFound, e)),
                }
            }
            ok.ok_or_else(|| Error::Io(last_err.unwrap()))?
        };
        let lib_bmrt = {
            let mut last_err: Option<std::io::Error> = None;
            let mut ok = None;
            for p in try_paths("libbmrt.so")
                .into_iter()
                .chain(try_paths("libbmrt.dylib"))
            {
                match unsafe { libloading::Library::new(&p) } {
                    Ok(l) => {
                        ok = Some(l);
                        break;
                    }
                    Err(e) => last_err = Some(std::io::Error::new(std::io::ErrorKind::NotFound, e)),
                }
            }
            ok.ok_or_else(|| Error::Io(last_err.unwrap()))?
        };
        Ok(Box::new(FfiBackend {
            _lib_bmlib: lib_bmlib,
            _lib_bmrt: lib_bmrt,
        }))
    }
    fn load_bmodel(&mut self, _path: &Path) -> Result<()> {
        // 预留：后续通过符号绑定 bmrt_create/bmrt_load_bmodel 等实现
        Err(Error::Backend(
            "当前环境未完成 FFI 符号绑定，后续可启用".into(),
        ))
    }
    fn get_networks(&self) -> Result<Vec<String>> {
        Err(Error::Backend(
            "当前环境未完成 FFI 符号绑定，后续可启用".into(),
        ))
    }
    fn infer(&self, _net_name: &str, _inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        Err(Error::Backend(
            "当前环境未完成 FFI 符号绑定，后续可启用".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mock_infer_roundtrip() {
        let mut rt = SophonRuntime::new_auto(0).unwrap();
        // 使用临时文件模拟 bmodel
        let tmp = tempfile::NamedTempFile::new().unwrap();
        rt.load_bmodel(tmp.path()).unwrap();
        let input = Tensor {
            dtype: DType::F32,
            shape: Shape::new(vec![1, 3, 28, 28]),
            data: vec![0u8; 1 * 3 * 28 * 28 * 4],
        };
        let outs = rt.infer("net", &[input]).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].shape.elements(), (1 * 3 * 28 * 28) as u64);
    }
}

