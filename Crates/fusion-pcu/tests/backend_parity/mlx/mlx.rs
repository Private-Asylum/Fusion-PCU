//! MLX executes the same full scalar contract as every other active provider.
//! These required gates stay pending until the actual MLX image closes each family.
use fusion_pcu::global::PcuBackendChoice;

macro_rules! gate {
    ($name:ident, $verify:path) => {
        #[test]
        #[ignore = "requires actual MLX GPU; required common scalar parity gate"]
        fn $name() {
            $verify(PcuBackendChoice::Mlx);
        }
    };
}
gate!(
    mlx_twenty_two_carrier_identity_contract,
    super::identity::verify
);
gate!(
    mlx_twenty_two_carrier_broadcast_contract,
    super::identity::broadcast::verify
);
gate!(mlx_fourteen_width_integer_contract, super::integer::verify);
gate!(
    mlx_fourteen_width_clamped_integer_contract,
    super::integer::clamped::verify
);
gate!(
    mlx_eight_width_div_rem_contract,
    super::integer::div_rem::verify
);
gate!(mlx_six_format_binary_contract, super::binary::verify);
gate!(mlx_six_format_unary_contract, super::unary::verify);
