шо
Ќ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.12v2.3.0-54-gfcc4b966f18з©

l5o/l5o_fc/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	®8*"
shared_namel5o/l5o_fc/kernel
x
%l5o/l5o_fc/kernel/Read/ReadVariableOpReadVariableOpl5o/l5o_fc/kernel*
_output_shapes
:	®8*
dtype0
v
l5o/l5o_fc/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:8* 
shared_namel5o/l5o_fc/bias
o
#l5o/l5o_fc/bias/Read/ReadVariableOpReadVariableOpl5o/l5o_fc/bias*
_output_shapes
:8*
dtype0

NoOpNoOp
а
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ы
valueСBО BЗ
Й
layer_with_weights-0
layer-0
regularization_losses
trainable_variables
	variables
	keras_api

signatures
|
_inbound_nodes

kernel
	bias

regularization_losses
trainable_variables
	variables
	keras_api
 

0
	1

0
	1
≠
regularization_losses
metrics

layers
non_trainable_variables
trainable_variables
layer_metrics
layer_regularization_losses
	variables
 
 
][
VARIABLE_VALUEl5o/l5o_fc/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEl5o/l5o_fc/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
	1

0
	1
≠

regularization_losses
layer_metrics
metrics
non_trainable_variables
trainable_variables

layers
layer_regularization_losses
	variables
 

0
 
 
 
 
 
 
 
 
Б
serving_default_l5o_fc_inputPlaceholder*(
_output_shapes
:€€€€€€€€€®*
dtype0*
shape:€€€€€€€€€®
ё
StatefulPartitionedCallStatefulPartitionedCallserving_default_l5o_fc_inputl5o/l5o_fc/kernell5o/l5o_fc/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference_signature_wrapper_47328
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
и
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%l5o/l5o_fc/kernel/Read/ReadVariableOp#l5o/l5o_fc/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *'
f"R 
__inference__traced_save_47414
ї
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamel5o/l5o_fc/kernell5o/l5o_fc/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__traced_restore_47430®Ф
ї
і
>__inference_l5o_layer_call_and_return_conditional_losses_47338

inputs)
%l5o_fc_matmul_readvariableop_resource*
&l5o_fc_biasadd_readvariableop_resource
identityИ£
l5o_fc/MatMul/ReadVariableOpReadVariableOp%l5o_fc_matmul_readvariableop_resource*
_output_shapes
:	®8*
dtype02
l5o_fc/MatMul/ReadVariableOpИ
l5o_fc/MatMulMatMulinputs$l5o_fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82
l5o_fc/MatMul°
l5o_fc/BiasAdd/ReadVariableOpReadVariableOp&l5o_fc_biasadd_readvariableop_resource*
_output_shapes
:8*
dtype02
l5o_fc/BiasAdd/ReadVariableOpЭ
l5o_fc/BiasAddBiasAddl5o_fc/MatMul:product:0%l5o_fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82
l5o_fc/BiasAddk
IdentityIdentityl5o_fc/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®:::P L
(
_output_shapes
:€€€€€€€€€®
 
_user_specified_nameinputs
д
~
#__inference_l5o_layer_call_fn_47317
l5o_fc_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCalll5o_fc_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_l5o_layer_call_and_return_conditional_losses_473102
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::22
StatefulPartitionedCallStatefulPartitionedCall:V R
(
_output_shapes
:€€€€€€€€€®
&
_user_specified_namel5o_fc_input
ƒ
®
>__inference_l5o_layer_call_and_return_conditional_losses_47280
l5o_fc_input
l5o_fc_47274
l5o_fc_47276
identityИҐl5o_fc/StatefulPartitionedCallН
l5o_fc/StatefulPartitionedCallStatefulPartitionedCalll5o_fc_inputl5o_fc_47274l5o_fc_47276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_l5o_fc_layer_call_and_return_conditional_losses_472542 
l5o_fc/StatefulPartitionedCallЬ
IdentityIdentity'l5o_fc/StatefulPartitionedCall:output:0^l5o_fc/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::2@
l5o_fc/StatefulPartitionedCalll5o_fc/StatefulPartitionedCall:V R
(
_output_shapes
:€€€€€€€€€®
&
_user_specified_namel5o_fc_input
ƒ
®
>__inference_l5o_layer_call_and_return_conditional_losses_47271
l5o_fc_input
l5o_fc_47265
l5o_fc_47267
identityИҐl5o_fc/StatefulPartitionedCallН
l5o_fc/StatefulPartitionedCallStatefulPartitionedCalll5o_fc_inputl5o_fc_47265l5o_fc_47267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_l5o_fc_layer_call_and_return_conditional_losses_472542 
l5o_fc/StatefulPartitionedCallЬ
IdentityIdentity'l5o_fc/StatefulPartitionedCall:output:0^l5o_fc/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::2@
l5o_fc/StatefulPartitionedCalll5o_fc/StatefulPartitionedCall:V R
(
_output_shapes
:€€€€€€€€€®
&
_user_specified_namel5o_fc_input
“
x
#__inference_l5o_layer_call_fn_47357

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_l5o_layer_call_and_return_conditional_losses_472922
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€®
 
_user_specified_nameinputs
Ў
{
&__inference_l5o_fc_layer_call_fn_47385

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_l5o_fc_layer_call_and_return_conditional_losses_472542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€®
 
_user_specified_nameinputs
≤
Ґ
>__inference_l5o_layer_call_and_return_conditional_losses_47310

inputs
l5o_fc_47304
l5o_fc_47306
identityИҐl5o_fc/StatefulPartitionedCallЗ
l5o_fc/StatefulPartitionedCallStatefulPartitionedCallinputsl5o_fc_47304l5o_fc_47306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_l5o_fc_layer_call_and_return_conditional_losses_472542 
l5o_fc/StatefulPartitionedCallЬ
IdentityIdentity'l5o_fc/StatefulPartitionedCall:output:0^l5o_fc/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::2@
l5o_fc/StatefulPartitionedCalll5o_fc/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€®
 
_user_specified_nameinputs
ь
њ
!__inference__traced_restore_47430
file_prefix&
"assignvariableop_l5o_l5o_fc_kernel&
"assignvariableop_1_l5o_l5o_fc_bias

identity_3ИҐAssignVariableOpҐAssignVariableOp_1Х
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*°
valueЧBФB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesФ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
RestoreV2/shape_and_slicesЇ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity°
AssignVariableOpAssignVariableOp"assignvariableop_l5o_l5o_fc_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp"assignvariableop_1_l5o_l5o_fc_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpР

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_2В

Identity_3IdentityIdentity_2:output:0^AssignVariableOp^AssignVariableOp_1*
T0*
_output_shapes
: 2

Identity_3"!

identity_3Identity_3:output:0*
_input_shapes

: ::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ќ
©
A__inference_l5o_fc_layer_call_and_return_conditional_losses_47376

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	®8*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:8*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®:::P L
(
_output_shapes
:€€€€€€€€€®
 
_user_specified_nameinputs
ї
і
>__inference_l5o_layer_call_and_return_conditional_losses_47348

inputs)
%l5o_fc_matmul_readvariableop_resource*
&l5o_fc_biasadd_readvariableop_resource
identityИ£
l5o_fc/MatMul/ReadVariableOpReadVariableOp%l5o_fc_matmul_readvariableop_resource*
_output_shapes
:	®8*
dtype02
l5o_fc/MatMul/ReadVariableOpИ
l5o_fc/MatMulMatMulinputs$l5o_fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82
l5o_fc/MatMul°
l5o_fc/BiasAdd/ReadVariableOpReadVariableOp&l5o_fc_biasadd_readvariableop_resource*
_output_shapes
:8*
dtype02
l5o_fc/BiasAdd/ReadVariableOpЭ
l5o_fc/BiasAddBiasAddl5o_fc/MatMul:product:0%l5o_fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82
l5o_fc/BiasAddk
IdentityIdentityl5o_fc/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®:::P L
(
_output_shapes
:€€€€€€€€€®
 
_user_specified_nameinputs
∆
~
#__inference_signature_wrapper_47328
l5o_fc_input
unknown
	unknown_0
identityИҐStatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCalll5o_fc_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__wrapped_model_472402
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::22
StatefulPartitionedCallStatefulPartitionedCall:V R
(
_output_shapes
:€€€€€€€€€®
&
_user_specified_namel5o_fc_input
≤
Ґ
>__inference_l5o_layer_call_and_return_conditional_losses_47292

inputs
l5o_fc_47286
l5o_fc_47288
identityИҐl5o_fc/StatefulPartitionedCallЗ
l5o_fc/StatefulPartitionedCallStatefulPartitionedCallinputsl5o_fc_47286l5o_fc_47288*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_l5o_fc_layer_call_and_return_conditional_losses_472542 
l5o_fc/StatefulPartitionedCallЬ
IdentityIdentity'l5o_fc/StatefulPartitionedCall:output:0^l5o_fc/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::2@
l5o_fc/StatefulPartitionedCalll5o_fc/StatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€®
 
_user_specified_nameinputs
“
x
#__inference_l5o_layer_call_fn_47366

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_l5o_layer_call_and_return_conditional_losses_473102
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€®
 
_user_specified_nameinputs
Д
Ќ
__inference__traced_save_47414
file_prefix0
,savev2_l5o_l5o_fc_kernel_read_readvariableop.
*savev2_l5o_l5o_fc_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_46e63cb2ef794a429349d509a03162ed/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameП
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*°
valueЧBФB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesО
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
SaveV2/shape_and_slicesЦ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_l5o_l5o_fc_kernel_read_readvariableop*savev2_l5o_l5o_fc_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*(
_input_shapes
: :	®8:8: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	®8: 

_output_shapes
:8:

_output_shapes
: 
Ќ
©
A__inference_l5o_fc_layer_call_and_return_conditional_losses_47254

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	®8*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:8*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®:::P L
(
_output_shapes
:€€€€€€€€€®
 
_user_specified_nameinputs
д
~
#__inference_l5o_layer_call_fn_47299
l5o_fc_input
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCalll5o_fc_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€8*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *G
fBR@
>__inference_l5o_layer_call_and_return_conditional_losses_472922
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®::22
StatefulPartitionedCallStatefulPartitionedCall:V R
(
_output_shapes
:€€€€€€€€€®
&
_user_specified_namel5o_fc_input
п
§
 __inference__wrapped_model_47240
l5o_fc_input-
)l5o_l5o_fc_matmul_readvariableop_resource.
*l5o_l5o_fc_biasadd_readvariableop_resource
identityИѓ
 l5o/l5o_fc/MatMul/ReadVariableOpReadVariableOp)l5o_l5o_fc_matmul_readvariableop_resource*
_output_shapes
:	®8*
dtype02"
 l5o/l5o_fc/MatMul/ReadVariableOpЪ
l5o/l5o_fc/MatMulMatMull5o_fc_input(l5o/l5o_fc/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82
l5o/l5o_fc/MatMul≠
!l5o/l5o_fc/BiasAdd/ReadVariableOpReadVariableOp*l5o_l5o_fc_biasadd_readvariableop_resource*
_output_shapes
:8*
dtype02#
!l5o/l5o_fc/BiasAdd/ReadVariableOp≠
l5o/l5o_fc/BiasAddBiasAddl5o/l5o_fc/MatMul:product:0)l5o/l5o_fc/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€82
l5o/l5o_fc/BiasAddo
IdentityIdentityl5o/l5o_fc/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€82

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€®:::V R
(
_output_shapes
:€€€€€€€€€®
&
_user_specified_namel5o_fc_input"ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*і
serving_default†
F
l5o_fc_input6
serving_default_l5o_fc_input:0€€€€€€€€€®:
l5o_fc0
StatefulPartitionedCall:0€€€€€€€€€8tensorflow/serving/predict:‘=
Ѓ
layer_with_weights-0
layer-0
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*&call_and_return_all_conditional_losses
__call__
_default_save_signature"Ћ
_tf_keras_sequentialђ{"class_name": "Sequential", "name": "l5o", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "l5o", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 168]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "l5o_fc_input"}}, {"class_name": "Dense", "config": {"name": "l5o_fc", "trainable": true, "dtype": "float32", "units": 56, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 168}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 168]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "l5o", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 168]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "l5o_fc_input"}}, {"class_name": "Dense", "config": {"name": "l5o_fc", "trainable": true, "dtype": "float32", "units": 56, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
Ж
_inbound_nodes

kernel
	bias

regularization_losses
trainable_variables
	variables
	keras_api
*&call_and_return_all_conditional_losses
__call__"Ќ
_tf_keras_layer≥{"class_name": "Dense", "name": "l5o_fc", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "l5o_fc", "trainable": true, "dtype": "float32", "units": 56, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 168}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 168]}}
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 
regularization_losses
metrics

layers
non_trainable_variables
trainable_variables
layer_metrics
layer_regularization_losses
	variables
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
 "
trackable_list_wrapper
$:"	®82l5o/l5o_fc/kernel
:82l5o/l5o_fc/bias
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
≠

regularization_losses
layer_metrics
metrics
non_trainable_variables
trainable_variables

layers
layer_regularization_losses
	variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∆2√
>__inference_l5o_layer_call_and_return_conditional_losses_47271
>__inference_l5o_layer_call_and_return_conditional_losses_47348
>__inference_l5o_layer_call_and_return_conditional_losses_47338
>__inference_l5o_layer_call_and_return_conditional_losses_47280ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Џ2„
#__inference_l5o_layer_call_fn_47366
#__inference_l5o_layer_call_fn_47317
#__inference_l5o_layer_call_fn_47299
#__inference_l5o_layer_call_fn_47357ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
д2б
 __inference__wrapped_model_47240Љ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *,Ґ)
'К$
l5o_fc_input€€€€€€€€€®
л2и
A__inference_l5o_fc_layer_call_and_return_conditional_losses_47376Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_l5o_fc_layer_call_fn_47385Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
7B5
#__inference_signature_wrapper_47328l5o_fc_inputС
 __inference__wrapped_model_47240m	6Ґ3
,Ґ)
'К$
l5o_fc_input€€€€€€€€€®
™ "/™,
*
l5o_fc К
l5o_fc€€€€€€€€€8Ґ
A__inference_l5o_fc_layer_call_and_return_conditional_losses_47376]	0Ґ-
&Ґ#
!К
inputs€€€€€€€€€®
™ "%Ґ"
К
0€€€€€€€€€8
Ъ z
&__inference_l5o_fc_layer_call_fn_47385P	0Ґ-
&Ґ#
!К
inputs€€€€€€€€€®
™ "К€€€€€€€€€8≠
>__inference_l5o_layer_call_and_return_conditional_losses_47271k	>Ґ;
4Ґ1
'К$
l5o_fc_input€€€€€€€€€®
p

 
™ "%Ґ"
К
0€€€€€€€€€8
Ъ ≠
>__inference_l5o_layer_call_and_return_conditional_losses_47280k	>Ґ;
4Ґ1
'К$
l5o_fc_input€€€€€€€€€®
p 

 
™ "%Ґ"
К
0€€€€€€€€€8
Ъ І
>__inference_l5o_layer_call_and_return_conditional_losses_47338e	8Ґ5
.Ґ+
!К
inputs€€€€€€€€€®
p

 
™ "%Ґ"
К
0€€€€€€€€€8
Ъ І
>__inference_l5o_layer_call_and_return_conditional_losses_47348e	8Ґ5
.Ґ+
!К
inputs€€€€€€€€€®
p 

 
™ "%Ґ"
К
0€€€€€€€€€8
Ъ Е
#__inference_l5o_layer_call_fn_47299^	>Ґ;
4Ґ1
'К$
l5o_fc_input€€€€€€€€€®
p

 
™ "К€€€€€€€€€8Е
#__inference_l5o_layer_call_fn_47317^	>Ґ;
4Ґ1
'К$
l5o_fc_input€€€€€€€€€®
p 

 
™ "К€€€€€€€€€8
#__inference_l5o_layer_call_fn_47357X	8Ґ5
.Ґ+
!К
inputs€€€€€€€€€®
p

 
™ "К€€€€€€€€€8
#__inference_l5o_layer_call_fn_47366X	8Ґ5
.Ґ+
!К
inputs€€€€€€€€€®
p 

 
™ "К€€€€€€€€€8§
#__inference_signature_wrapper_47328}	FҐC
Ґ 
<™9
7
l5o_fc_input'К$
l5o_fc_input€€€€€€€€€®"/™,
*
l5o_fc К
l5o_fc€€€€€€€€€8