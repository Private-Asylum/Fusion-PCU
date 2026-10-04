//! External compiler/validator evidence only; this test never opens a device.
use pcu_facade::PcuScalarType;
use std::sync::atomic::{AtomicUsize, Ordering};
static FILES: AtomicUsize = AtomicUsize::new(0);
type NativeResult<T> = Result<T, Box<dyn std::error::Error>>;
#[path = "../../benches/composed_float_maps/ffi/compile/compile.rs"]
mod compile;

fn validate(words: &[u32], index: usize) {
    assert_eq!(words[0], 0x0723_0203);
    let mut offset = 5;
    while offset < words.len() {
        let count = usize::try_from(words[offset] >> 16).unwrap();
        assert!(count > 0 && offset + count <= words.len());
        if words[offset] & 0xffff == 17 {
            // Only Shader capability; no native Float64/Int64 arithmetic dependency.
            assert_eq!(&words[offset + 1..offset + count], &[1]);
        }
        offset += count;
    }
    let path = std::env::temp_dir().join(format!(
        "pcu-native-composed-validation-{}-{index}-{}.spv",
        std::process::id(),
        FILES.fetch_add(1, Ordering::Relaxed)
    ));
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    std::fs::write(&path, bytes).unwrap();
    let result = std::process::Command::new("spirv-val")
        .args(["--target-env", "vulkan1.0"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{} {}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
#[ignore = "requires installed GLSL compiler and official SPIR-V validator; no GPU"]
fn independent_native_glsl_compiler_validates_all_formats_and_policies() {
    all_modules(false);
}

#[test]
#[ignore = "requires installed GLSL compiler and official validator; independent one-effect control"]
fn independent_one_effect_glsl_compiler_validates_all_formats_and_policies() {
    all_modules(true);
}

fn all_modules(one_effect: bool) {
    let mut index = 0;
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        for policy in 0..3 {
            for range in 0..2 {
                for extent in [1, 65] {
                    let words = compile::shader(policy, range, scalar, extent, one_effect).unwrap();
                    validate(&words, index);
                    index += 1;
                }
            }
        }
    }
    assert_eq!(index, 72);
}

#[test]
#[ignore = "requires installed GLSL compiler and official validator; independent integer control"]
fn independent_integer_glsl_compiler_validates_all_widths_and_packed_extents() {
    let mut index = 0;
    for scalar in [
        PcuScalarType::U8,
        PcuScalarType::I8,
        PcuScalarType::U16,
        PcuScalarType::I16,
        PcuScalarType::U32,
        PcuScalarType::I32,
        PcuScalarType::U64,
        PcuScalarType::I64,
        PcuScalarType::U128,
        PcuScalarType::I128,
        PcuScalarType::U256,
        PcuScalarType::I256,
        PcuScalarType::U512,
        PcuScalarType::I512,
    ] {
        for range in 0..2 {
            for extent in [1, 3, 4, 5, 65] {
                let words = compile::shader(0, range, scalar, extent, false).unwrap();
                validate(&words, index);
                index += 1;
            }
        }
    }
    assert_eq!(index, 140);
}

#[test]
#[ignore = "requires installed GLSL compiler and official validator; discarded integer effect"]
fn independent_discarded_integer_effect_validates_packed_extents() {
    let mut index = 0;
    for scalar in [
        PcuScalarType::U8,
        PcuScalarType::I8,
        PcuScalarType::U16,
        PcuScalarType::I16,
        PcuScalarType::U32,
        PcuScalarType::I32,
        PcuScalarType::U64,
        PcuScalarType::I64,
        PcuScalarType::U128,
        PcuScalarType::I128,
        PcuScalarType::U256,
        PcuScalarType::I256,
        PcuScalarType::U512,
        PcuScalarType::I512,
    ] {
        for range in 0..2 {
            for extent in [1, 3, 4, 5, 65] {
                validate(
                    &compile::shader(0, range, scalar, extent, true).unwrap(),
                    index,
                );
                index += 1;
            }
        }
    }
    assert_eq!(index, 140);
}

#[cfg(feature = "insights")]
#[allow(dead_code)] // This pure test exercises the shared handwritten owner's attachment seam.
#[path = "../../benches/composed_float_maps/ffi/api_census/api_census.rs"]
mod api_census;
#[cfg(feature = "insights")]
#[test]
fn handwritten_sdk_attachment_restores_nested_unwind_and_retains_origin() {
    use fusion_pcu_vulkan::{PcuVulkanApiInsights, PcuVulkanApiPoint as Point, PcuVulkanCountOnlyClock};
    use std::rc::Rc;
    let first = Rc::new(PcuVulkanApiInsights::new(PcuVulkanCountOnlyClock));
    let other = Rc::new(PcuVulkanApiInsights::new(PcuVulkanCountOnlyClock));
    assert!(api_census::current().is_none());
    api_census::with_scope(Rc::clone(&first), || {
        let owner = api_census::current().unwrap();
        api_census::count_current(Point::CreateDevice);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            api_census::with_scope(Rc::clone(&other), || {
                api_census::count_current(Point::CreateDevice);
                api_census::count(Some(&owner), Point::QueueSubmit);
                panic!("explicit attachment unwind witness");
            });
        }));
        assert!(result.is_err());
        assert!(Rc::ptr_eq(&api_census::current().unwrap(), &first));
        api_census::count_current(Point::DestroyDevice);
    });
    assert!(api_census::current().is_none());
    assert_eq!(Rc::strong_count(&first), 1);
    assert_eq!(Rc::strong_count(&other), 1);
    let a = first.records();
    let b = other.records();
    assert_eq!(a[Point::CreateDevice.index()].count, 1);
    assert_eq!(a[Point::DestroyDevice.index()].count, 1);
    assert_eq!(a[Point::QueueSubmit.index()].count, 1);
    assert_eq!(b[Point::CreateDevice.index()].count, 1);
    assert_eq!(b[Point::QueueSubmit.index()].count, 0);
    for record in a.into_iter().chain(b) {
        assert_eq!(
            (record.hits, record.inclusive_ticks, record.exclusive_ticks),
            (0, 0, 0)
        );
    }
}
