# Prepared CPU DivRem scope

`PcuCpuCheckedDivRem<T>` admits canonical checked quotient/remainder maps for all fourteen
sealed integer carriers: signed and unsigned 8, 16, 32, 64, 128, 256 and 512 bits.
Cold preparation freezes four complete bindings, logical extent, actual SSA operand mapping,
exact representation and a monomorphized executable. The unified host route uses four arguments.

Direct and canonical grid-stride maps execute logical invocations in ascending order.
Calls validate both input and both output spans, preflight only zero and signed MIN/-1 domains,
then publish both output prefixes. A fatal fault preserves both outputs and tails; retry
reuses the same plan. No caller pointer or IR survives a call. The warm path allocates
no storage and performs no discovery, ranking or graph evaluation. Wide division computes one joint limb division per lane during publication.
Sealed same-width integers have no other quotient/remainder range fault, so the preceding domain
scan performs no division. This is an executor realization change, not a timing claim.

Division truncates toward zero; signed remainder retains the dividend's sign. Zero reports
`DivideByZero`; signed MIN/-1 reports `SignedDivisionOverflow` for the joint operation.
This is the exact PCU integer contract, with no floating arithmetic or environment dependency.
Legacy eight-width IDs128..135 remain numerically unchanged; wide IDs512..517 remain unchanged. All14 now use executor revision2, zero workspace.

The legacy profile remains Reject/Unspecified, indexed direct/grid loads, four bindings and empty flags.
Boundary/Strict and compound/precision permissions preserve the same stronger checked integer
semantics; all three floating underflow policies are carried unchanged and irrelevant to integer
arithmetic. Portable DivRem, Clamp and total divide-or-zero flags remain separate unproved policies.

Independent proof fixtures compare complete native-endian carrier values to a base-256 long
division oracle qualified against Node BigInt text goldens. These tests and genuine source,
prepared source, explicit graph and independent native controls are separate from older eight-width
proofs. Linux/M4 native qualification of this fourteen-width cut is recorded in the CPU plan.

A separately identified operand-role profile admits three/four declarations, actual repeated reads,
read-only scalar/index-zero broadcasts, unread read-only declarations, and reordered destinations.
IDs8192..8205 revision1 are selected only when the original canonical validator refuses and the
new actual-operand assessor succeeds. Original declarations retain exact type/access validation;
proved unread spans require zero elements, scalar reads one, and indexed reads the logical extent.
The cold table freezes quotient/remainder destinations and four const-index executable choices.
Each distinct read resource is borrowed once; repeated operands retain that immutable borrow.
Whole-map domain preflight and terminal dual publication remain the same transaction. This
profile has independent genuine source/graph/native fixtures; its native certificate is separate
from the earlier canonical wide and domain-preflight certificates.
