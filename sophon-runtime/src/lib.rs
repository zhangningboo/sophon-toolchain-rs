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

#[derive(Debug, Clone)]
pub struct IoDesc {
    pub name: String,
    pub dtype: DType,
    pub scale: f32,
    pub shape: Shape,
}

#[derive(Debug, Clone)]
pub struct NetInfo {
    pub name: String,
    pub input_num: i32,
    pub output_num: i32,
    pub inputs: Vec<IoDesc>,
    pub outputs: Vec<IoDesc>,
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("backend error: {0}")]
    Backend(String),
    #[error("invalid argument: {0}")]
    Invalid(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("ffi error: {0}")]
    Ffi(String),
}

pub type Result<T> = std::result::Result<T, Error>;

pub trait RuntimeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn create(devid: i32) -> Result<Box<dyn RuntimeBackend>>
    where
        Self: Sized;
    fn load_bmodel(&mut self, path: &Path) -> Result<()>;
    fn get_networks(&self) -> Result<Vec<String>>;
    fn net_info(&self, net_name: &str) -> Result<NetInfo>;
    fn infer(&self, net_name: &str, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
}

pub struct SophonRuntime {
    backend: Box<dyn RuntimeBackend>,
}

impl SophonRuntime {
    pub fn new_auto(devid: i32) -> Result<Self> {
        #[cfg(feature = "ffi")]
        {
            let b = FfiBackend::create(devid)?;
            Ok(Self { backend: b })
        }
        #[cfg(not(feature = "ffi"))]
        {
            Err(Error::Backend("未启用 FFI 后端".into()))
        }
    }
    pub fn load_bmodel(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.backend.load_bmodel(path.as_ref())
    }
    pub fn networks(&self) -> Result<Vec<String>> {
        self.backend.get_networks()
    }
    pub fn net_info(&self, net_name: &str) -> Result<NetInfo> {
        self.backend.net_info(net_name)
    }
    pub fn infer(&self, net_name: &str, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.backend.infer(net_name, inputs)
    }
    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }
}
#[cfg(feature = "ffi")]
pub fn available_devices(max_probe: i32) -> Result<Vec<i32>> {
    let mut v = Vec::new();
    for id in 0..max_probe {
        match FfiBackend::create(id) {
            Ok(b) => {
                v.push(id);
                drop(b);
            }
            Err(_) => {}
        }
    }
    Ok(v)
}

#[cfg(not(feature = "ffi"))]
pub fn available_devices(_max_probe: i32) -> Result<Vec<i32>> {
    Ok(vec![])
}

#[cfg(feature = "mock")]
struct MockBackend;

#[cfg(feature = "mock")]
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
    fn net_info(&self, net_name: &str) -> Result<NetInfo> {
        Ok(NetInfo {
            name: net_name.to_string(),
            input_num: 1,
            output_num: 1,
            inputs: vec![IoDesc {
                name: "input".to_string(),
                dtype: DType::F32,
                scale: 1.0,
                shape: Shape::new(vec![1, 3, 640, 640]),
            }],
            outputs: vec![IoDesc {
                name: "output".to_string(),
                dtype: DType::F32,
                scale: 1.0,
                shape: Shape::new(vec![1, 8400, 84]),
            }],
        })
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

#[cfg(feature = "ffi")]
struct FfiBackend {
    lib_bmlib: libloading::Library,
    lib_bmrt: libloading::Library,
    bm_dev_request: unsafe extern "C" fn(handle: *mut *mut std::ffi::c_void, devid: std::os::raw::c_int) -> std::os::raw::c_int,
    bm_dev_free: unsafe extern "C" fn(handle: *mut std::ffi::c_void),
    bm_thread_sync: unsafe extern "C" fn(handle: *mut std::ffi::c_void) -> std::os::raw::c_int,
    bmrt_create: unsafe extern "C" fn(handle: *mut std::ffi::c_void) -> *mut std::ffi::c_void,
    bmrt_destroy: unsafe extern "C" fn(p_bmrt: *mut std::ffi::c_void),
    bmrt_load_bmodel: unsafe extern "C" fn(p_bmrt: *mut std::ffi::c_void, path: *const std::os::raw::c_char) -> bool,
    bmrt_get_network_number: unsafe extern "C" fn(p_bmrt: *mut std::ffi::c_void) -> std::os::raw::c_int,
    bmrt_get_network_names: unsafe extern "C" fn(p_bmrt: *mut std::ffi::c_void, network_names: *mut *mut *const std::os::raw::c_char),
    bmrt_get_network_info: unsafe extern "C" fn(p_bmrt: *mut std::ffi::c_void, net_name: *const std::os::raw::c_char) -> *const BmNetInfo,
    bmrt_launch_data: unsafe extern "C" fn(
        p_bmrt: *mut std::ffi::c_void,
        net_name: *const std::os::raw::c_char,
        input_datas: *const *const std::ffi::c_void,
        input_shapes: *const BmShape,
        input_num: std::os::raw::c_int,
        output_datas: *mut *mut std::ffi::c_void,
        output_shapes: *mut BmShape,
        output_num: std::os::raw::c_int,
        user_mem: bool,
    ) -> bool,
    bm_handle: *mut std::ffi::c_void,
    p_bmrt: *mut std::ffi::c_void,
}

#[cfg(feature = "ffi")]
unsafe impl Send for FfiBackend {}
#[cfg(feature = "ffi")]
unsafe impl Sync for FfiBackend {}
#[repr(C)]
#[derive(Clone, Copy)]
struct BmShape {
    num_dims: std::os::raw::c_int,
    dims: [std::os::raw::c_int; 8],
}

#[repr(C)]
struct BmStageInfo {
    input_shapes: *mut BmShape,
    output_shapes: *mut BmShape,
}

#[repr(C)]
struct BmNetInfo {
    name: *const std::os::raw::c_char,
    is_dynamic: bool,
    input_num: std::os::raw::c_int,
    input_names: *mut *const std::os::raw::c_char,
    input_dtypes: *mut std::os::raw::c_int,
    input_scales: *mut f32,
    output_num: std::os::raw::c_int,
    output_names: *mut *const std::os::raw::c_char,
    output_dtypes: *mut std::os::raw::c_int,
    output_scales: *mut f32,
    stage_num: std::os::raw::c_int,
    stages: *mut BmStageInfo,
    max_input_bytes: *mut usize,
    max_output_bytes: *mut usize,
}

fn dtype_bytes(d: i32) -> usize {
    match d {
        0 => 4,
        1 => 2,
        2 => 1,
        3 => 1,
        4 => 2,
        5 => 2,
        6 => 4,
        7 => 4,
        _ => 1,
    }
}

fn dtype_map(d: i32) -> DType {
    match d {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::I8,
        3 => DType::U8,
        4 => DType::I16,
        5 => DType::U16,
        6 => DType::I32,
        7 => DType::U32,
        _ => DType::U8,
    }
}

#[cfg(feature = "ffi")]
impl RuntimeBackend for FfiBackend {
    fn name(&self) -> &'static str {
        "ffi"
    }
    fn create(devid: i32) -> Result<Box<dyn RuntimeBackend>>
    where
        Self: Sized,
    {
        let libdir = std::env::var("SOPHON_SDK_LIBDIR").ok();
        let try_paths = |name: &str| -> Vec<std::path::PathBuf> {
            let mut v = Vec::new();
            if let Some(dir) = &libdir {
                v.push(Path::new(dir).join(name));
            }
            v.push(Path::new("/opt/sophon/libsophon-current/lib").join(name));
            v.push(Path::new("/opt/sophon/lib").join(name));
            v.push(Path::new(name).to_path_buf());
            v
        };
        let lib_bmlib = {
            let mut last_err: Option<std::io::Error> = None;
            let mut ok = None;
            for p in try_paths("libbmlib.so")
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
        unsafe {
            let (
                bm_dev_request,
                bm_dev_free,
                bm_thread_sync,
                bmrt_create,
                bmrt_destroy,
                bmrt_load_bmodel,
                bmrt_get_network_number,
                bmrt_get_network_names,
                bmrt_get_network_info,
                bmrt_launch_data,
            ) = {
                let s_bm_dev_request: libloading::Symbol<unsafe extern "C" fn(*mut *mut std::ffi::c_void, std::os::raw::c_int) -> std::os::raw::c_int> =
                    lib_bmlib.get(b"bm_dev_request").map_err(|e| Error::Ffi(e.to_string()))?;
                let s_bm_dev_free: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> =
                    lib_bmlib.get(b"bm_dev_free").map_err(|e| Error::Ffi(e.to_string()))?;
                let s_bm_thread_sync: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void) -> std::os::raw::c_int> =
                    lib_bmlib.get(b"bm_thread_sync").map_err(|e| Error::Ffi(e.to_string()))?;
                let s_bmrt_create: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut std::ffi::c_void> =
                    lib_bmrt.get(b"bmrt_create").map_err(|e| Error::Ffi(e.to_string()))?;
                let s_bmrt_destroy: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void)> =
                    lib_bmrt.get(b"bmrt_destroy").map_err(|e| Error::Ffi(e.to_string()))?;
                let s_bmrt_load_bmodel: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *const std::os::raw::c_char) -> bool> =
                    lib_bmrt.get(b"bmrt_load_bmodel").map_err(|e| Error::Ffi(e.to_string()))?;
                let s_bmrt_get_network_number: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void) -> std::os::raw::c_int> =
                    lib_bmrt.get(b"bmrt_get_network_number").map_err(|e| Error::Ffi(e.to_string()))?;
                let s_bmrt_get_network_names: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *mut *mut *const std::os::raw::c_char)> =
                    lib_bmrt.get(b"bmrt_get_network_names").map_err(|e| Error::Ffi(e.to_string()))?;
                let s_bmrt_get_network_info: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *const std::os::raw::c_char) -> *const BmNetInfo> =
                    lib_bmrt.get(b"bmrt_get_network_info").map_err(|e| Error::Ffi(e.to_string()))?;
                let s_bmrt_launch_data: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *const std::os::raw::c_char, *const *const std::ffi::c_void, *const BmShape, std::os::raw::c_int, *mut *mut std::ffi::c_void, *mut BmShape, std::os::raw::c_int, bool) -> bool> =
                    lib_bmrt.get(b"bmrt_launch_data").map_err(|e| Error::Ffi(e.to_string()))?;
                (
                    *s_bm_dev_request,
                    *s_bm_dev_free,
                    *s_bm_thread_sync,
                    *s_bmrt_create,
                    *s_bmrt_destroy,
                    *s_bmrt_load_bmodel,
                    *s_bmrt_get_network_number,
                    *s_bmrt_get_network_names,
                    *s_bmrt_get_network_info,
                    *s_bmrt_launch_data,
                )
            };
            let mut handle: *mut std::ffi::c_void = std::ptr::null_mut();
            let s = bm_dev_request(&mut handle, devid);
            if s != 0 {
                return Err(Error::Backend(format!("bm_dev_request 失败(code: {})，请检查驱动是否已安装并启用、/dev/bmdev* 是否存在、权限是否足够", s)));
            }
            let p_bmrt = bmrt_create(handle);
            if p_bmrt.is_null() {
                bm_dev_free(handle);
                return Err(Error::Backend("bmrt_create 失败".into()));
            }
            Ok(Box::new(FfiBackend {
                lib_bmlib,
                lib_bmrt,
                bm_dev_request: bm_dev_request,
                bm_dev_free: bm_dev_free,
                bm_thread_sync: bm_thread_sync,
                bmrt_create: bmrt_create,
                bmrt_destroy: bmrt_destroy,
                bmrt_load_bmodel: bmrt_load_bmodel,
                bmrt_get_network_number: bmrt_get_network_number,
                bmrt_get_network_names: bmrt_get_network_names,
                bmrt_get_network_info: bmrt_get_network_info,
                bmrt_launch_data: bmrt_launch_data,
                bm_handle: handle,
                p_bmrt,
            }))
        }
    }
    fn load_bmodel(&mut self, path: &Path) -> Result<()> {
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes()).map_err(|e| Error::Invalid(e.to_string()))?;
        let ok = unsafe { (self.bmrt_load_bmodel)(self.p_bmrt, c_path.as_ptr()) };
        if ok { Ok(()) } else { Err(Error::Backend("bmrt_load_bmodel 失败".into())) }
    }
    fn get_networks(&self) -> Result<Vec<String>> {
        let num = unsafe { (self.bmrt_get_network_number)(self.p_bmrt) };
        if num <= 0 {
            return Ok(Vec::new());
        }
        let mut networks: *mut *const std::os::raw::c_char = std::ptr::null_mut();
        unsafe { (self.bmrt_get_network_names)(self.p_bmrt, &mut networks) };
        if networks.is_null() {
            return Ok(Vec::new());
        }
        let slice = unsafe { std::slice::from_raw_parts(networks, num as usize) };
        let mut out = Vec::with_capacity(num as usize);
        for &p in slice {
            if p.is_null() {
                continue;
            }
            let s = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned();
            out.push(s);
        }
        unsafe { libc::free(networks as *mut libc::c_void) };
        Ok(out)
    }
    fn net_info(&self, net_name: &str) -> Result<NetInfo> {
        let c_name = std::ffi::CString::new(net_name.as_bytes())
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let net_info_ptr = unsafe { (self.bmrt_get_network_info)(self.p_bmrt, c_name.as_ptr()) };
        if net_info_ptr.is_null() {
            return Err(Error::Backend("bmrt_get_network_info 失败".into()));
        }
        let net_info = unsafe { &*net_info_ptr };
        if net_info.stage_num <= 0 || net_info.stages.is_null() {
            return Err(Error::Backend("network stage 为空".into()));
        }
        let stages =
            unsafe { std::slice::from_raw_parts(net_info.stages, net_info.stage_num as usize) };
        let stage0 = &stages[0];

        let input_names =
            unsafe { std::slice::from_raw_parts(net_info.input_names, net_info.input_num as usize) };
        let input_dtypes = unsafe {
            std::slice::from_raw_parts(net_info.input_dtypes, net_info.input_num as usize)
        };
        let input_scales = unsafe {
            std::slice::from_raw_parts(net_info.input_scales, net_info.input_num as usize)
        };
        let input_shapes =
            unsafe { std::slice::from_raw_parts(stage0.input_shapes, net_info.input_num as usize) };

        let mut inputs = Vec::with_capacity(net_info.input_num as usize);
        for i in 0..(net_info.input_num as usize) {
            let name = if input_names[i].is_null() {
                format!("input_{}", i)
            } else {
                unsafe { std::ffi::CStr::from_ptr(input_names[i]) }
                    .to_string_lossy()
                    .into_owned()
            };
            let shape = {
                let s = &input_shapes[i];
                let mut v = Vec::with_capacity(s.num_dims as usize);
                for j in 0..(s.num_dims as usize) {
                    v.push(s.dims[j]);
                }
                Shape { dims: v }
            };
            inputs.push(IoDesc {
                name,
                dtype: dtype_map(input_dtypes[i]),
                scale: input_scales[i],
                shape,
            });
        }

        let output_names = unsafe {
            std::slice::from_raw_parts(net_info.output_names, net_info.output_num as usize)
        };
        let output_dtypes = unsafe {
            std::slice::from_raw_parts(net_info.output_dtypes, net_info.output_num as usize)
        };
        let output_scales = unsafe {
            std::slice::from_raw_parts(net_info.output_scales, net_info.output_num as usize)
        };
        let output_shapes = unsafe {
            std::slice::from_raw_parts(stage0.output_shapes, net_info.output_num as usize)
        };

        let mut outputs = Vec::with_capacity(net_info.output_num as usize);
        for i in 0..(net_info.output_num as usize) {
            let name = if output_names[i].is_null() {
                format!("output_{}", i)
            } else {
                unsafe { std::ffi::CStr::from_ptr(output_names[i]) }
                    .to_string_lossy()
                    .into_owned()
            };
            let shape = {
                let s = &output_shapes[i];
                let mut v = Vec::with_capacity(s.num_dims as usize);
                for j in 0..(s.num_dims as usize) {
                    v.push(s.dims[j]);
                }
                Shape { dims: v }
            };
            outputs.push(IoDesc {
                name,
                dtype: dtype_map(output_dtypes[i]),
                scale: output_scales[i],
                shape,
            });
        }

        Ok(NetInfo {
            name: net_name.to_string(),
            input_num: net_info.input_num,
            output_num: net_info.output_num,
            inputs,
            outputs,
        })
    }
    fn infer(&self, net_name: &str, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(Error::Invalid("输入为空".into()));
        }
        let c_name = std::ffi::CString::new(net_name.as_bytes())
            .map_err(|e| Error::Invalid(e.to_string()))?;
        let net_info_ptr = unsafe { (self.bmrt_get_network_info)(self.p_bmrt, c_name.as_ptr()) };
        if net_info_ptr.is_null() {
            return Err(Error::Backend("bmrt_get_network_info 失败".into()));
        }
        let net_info = unsafe { &*net_info_ptr };
        let input_num = inputs.len() as std::os::raw::c_int;
        let mut in_ptrs: Vec<*const std::ffi::c_void> = Vec::with_capacity(inputs.len());
        let mut in_shapes: Vec<BmShape> = Vec::with_capacity(inputs.len());
        for t in inputs {
            in_ptrs.push(t.data.as_ptr() as *const std::ffi::c_void);
            let mut dims = [0i32; 8];
            for (i, d) in t.shape.dims.iter().enumerate().take(8) {
                dims[i] = *d;
            }
            in_shapes.push(BmShape {
                num_dims: t.shape.dims.len() as i32,
                dims,
            });
        }
        let output_num = net_info.output_num;
        let mut out_ptrs: Vec<*mut std::ffi::c_void> =
            vec![std::ptr::null_mut(); output_num as usize];
        let mut out_shapes: Vec<BmShape> = vec![
            BmShape {
                num_dims: 0,
                dims: [0; 8],
            };
            output_num as usize
        ];
        let ok = unsafe {
            (self.bmrt_launch_data)(
                self.p_bmrt,
                c_name.as_ptr(),
                in_ptrs.as_ptr(),
                in_shapes.as_ptr(),
                input_num,
                out_ptrs.as_mut_ptr(),
                out_shapes.as_mut_ptr(),
                output_num,
                false,
            )
        };
        if !ok {
            return Err(Error::Backend("bmrt_launch_data 失败".into()));
        }
        let dtypes_slice =
            unsafe { std::slice::from_raw_parts(net_info.output_dtypes, output_num as usize) };
        let mut outs = Vec::with_capacity(output_num as usize);
        for i in 0..(output_num as usize) {
            let shape = {
                let s = &out_shapes[i];
                let mut v = Vec::with_capacity(s.num_dims as usize);
                for j in 0..(s.num_dims as usize) {
                    v.push(s.dims[j]);
                }
                Shape { dims: v }
            };
            let bytes_per_elem = dtype_bytes(dtypes_slice[i]);
            let total_elems = shape.elements() as usize;
            let total_bytes = total_elems * bytes_per_elem;
            let ptr = out_ptrs[i];
            if ptr.is_null() {
                return Err(Error::Backend("输出指针为空".into()));
            }
            let data =
                unsafe { std::slice::from_raw_parts(ptr as *const u8, total_bytes) }.to_vec();
            unsafe { libc::free(ptr as *mut libc::c_void) };
            outs.push(Tensor {
                dtype: dtype_map(dtypes_slice[i]),
                shape,
                data,
            });
        }
        Ok(outs)
    }
}

#[cfg(feature = "ffi")]
impl Drop for FfiBackend {
    fn drop(&mut self) {
        unsafe {
            (self.bmrt_destroy)(self.p_bmrt);
            (self.bm_dev_free)(self.bm_handle);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shape_elements() {
        assert_eq!(Shape::new(Vec::<i32>::new()).elements(), 1);
        assert_eq!(
            Shape::new(vec![1, 3, 28, 28]).elements(),
            (1 * 3 * 28 * 28) as u64
        );
    }
}
