зЋ
ЭЃ
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
dtypetype
О
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ыв
 
"l4_integration/l2_integ_fc0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:AB*3
shared_name$"l4_integration/l2_integ_fc0/kernel

6l4_integration/l2_integ_fc0/kernel/Read/ReadVariableOpReadVariableOp"l4_integration/l2_integ_fc0/kernel*
_output_shapes

:AB*
dtype0

 l4_integration/l2_integ_fc0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:B*1
shared_name" l4_integration/l2_integ_fc0/bias

4l4_integration/l2_integ_fc0/bias/Read/ReadVariableOpReadVariableOp l4_integration/l2_integ_fc0/bias*
_output_shapes
:B*
dtype0

NoOpNoOp
Р

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ћ	
valueё	Bю	 Bч	

layer_with_weights-0
layer-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
|
_inbound_nodes

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
f
_inbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
 

	0

1

	0

1
­
regularization_losses
metrics

layers
non_trainable_variables
trainable_variables
layer_metrics
layer_regularization_losses
	variables
 
 
nl
VARIABLE_VALUE"l4_integration/l2_integ_fc0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE l4_integration/l2_integ_fc0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
­
regularization_losses
layer_metrics
metrics
non_trainable_variables
trainable_variables

layers
layer_regularization_losses
	variables
 
 
 
 
­
regularization_losses
layer_metrics
metrics
 non_trainable_variables
trainable_variables

!layers
"layer_regularization_losses
	variables
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 

"serving_default_l2_integ_fc0_inputPlaceholder*'
_output_shapes
:џџџџџџџџџA*
dtype0*
shape:џџџџџџџџџA

StatefulPartitionedCallStatefulPartitionedCall"serving_default_l2_integ_fc0_input"l4_integration/l2_integ_fc0/kernel l4_integration/l2_integ_fc0/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_46080
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6l4_integration/l2_integ_fc0/kernel/Read/ReadVariableOp4l4_integration/l2_integ_fc0/bias/Read/ReadVariableOpConst*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_46178
н
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"l4_integration/l2_integ_fc0/kernel l4_integration/l2_integ_fc0/bias*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_46194вИ

П
I__inference_l4_integration_layer_call_and_return_conditional_losses_46062

inputs
l2_integ_fc0_46055
l2_integ_fc0_46057
identityЂ$l2_integ_fc0/StatefulPartitionedCallЅ
$l2_integ_fc0/StatefulPartitionedCallStatefulPartitionedCallinputsl2_integ_fc0_46055l2_integ_fc0_46057*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l2_integ_fc0_layer_call_and_return_conditional_losses_459902&
$l2_integ_fc0/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall-l2_integ_fc0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_460112
activation_19/PartitionedCallЁ
IdentityIdentity&activation_19/PartitionedCall:output:0%^l2_integ_fc0/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::2L
$l2_integ_fc0/StatefulPartitionedCall$l2_integ_fc0/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџA
 
_user_specified_nameinputs
ч

.__inference_l4_integration_layer_call_fn_46111

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_l4_integration_layer_call_and_return_conditional_losses_460432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџA
 
_user_specified_nameinputs
Ќ
Ы
I__inference_l4_integration_layer_call_and_return_conditional_losses_46030
l2_integ_fc0_input
l2_integ_fc0_46023
l2_integ_fc0_46025
identityЂ$l2_integ_fc0/StatefulPartitionedCallБ
$l2_integ_fc0/StatefulPartitionedCallStatefulPartitionedCalll2_integ_fc0_inputl2_integ_fc0_46023l2_integ_fc0_46025*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l2_integ_fc0_layer_call_and_return_conditional_losses_459902&
$l2_integ_fc0/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall-l2_integ_fc0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_460112
activation_19/PartitionedCallЁ
IdentityIdentity&activation_19/PartitionedCall:output:0%^l2_integ_fc0/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::2L
$l2_integ_fc0/StatefulPartitionedCall$l2_integ_fc0/StatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџA
,
_user_specified_namel2_integ_fc0_input
а
Џ
G__inference_l2_integ_fc0_layer_call_and_return_conditional_losses_46130

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:AB*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:B*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA:::O K
'
_output_shapes
:џџџџџџџџџA
 
_user_specified_nameinputs
а
Џ
G__inference_l2_integ_fc0_layer_call_and_return_conditional_losses_45990

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:AB*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:B*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA:::O K
'
_output_shapes
:џџџџџџџџџA
 
_user_specified_nameinputs
 

Ы
I__inference_l4_integration_layer_call_and_return_conditional_losses_46102

inputs/
+l2_integ_fc0_matmul_readvariableop_resource0
,l2_integ_fc0_biasadd_readvariableop_resource
identityД
"l2_integ_fc0/MatMul/ReadVariableOpReadVariableOp+l2_integ_fc0_matmul_readvariableop_resource*
_output_shapes

:AB*
dtype02$
"l2_integ_fc0/MatMul/ReadVariableOp
l2_integ_fc0/MatMulMatMulinputs*l2_integ_fc0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2
l2_integ_fc0/MatMulГ
#l2_integ_fc0/BiasAdd/ReadVariableOpReadVariableOp,l2_integ_fc0_biasadd_readvariableop_resource*
_output_shapes
:B*
dtype02%
#l2_integ_fc0/BiasAdd/ReadVariableOpЕ
l2_integ_fc0/BiasAddBiasAddl2_integ_fc0/MatMul:product:0+l2_integ_fc0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2
l2_integ_fc0/BiasAdd
activation_19/TanhTanhl2_integ_fc0/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџB2
activation_19/Tanhj
IdentityIdentityactivation_19/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA:::O K
'
_output_shapes
:џџџџџџџџџA
 
_user_specified_nameinputs
з

#__inference_signature_wrapper_46080
l2_integ_fc0_input
unknown
	unknown_0
identityЂStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCalll2_integ_fc0_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_459762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџA
,
_user_specified_namel2_integ_fc0_input


.__inference_l4_integration_layer_call_fn_46069
l2_integ_fc0_input
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalll2_integ_fc0_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_l4_integration_layer_call_and_return_conditional_losses_460622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџA
,
_user_specified_namel2_integ_fc0_input

П
I__inference_l4_integration_layer_call_and_return_conditional_losses_46043

inputs
l2_integ_fc0_46036
l2_integ_fc0_46038
identityЂ$l2_integ_fc0/StatefulPartitionedCallЅ
$l2_integ_fc0/StatefulPartitionedCallStatefulPartitionedCallinputsl2_integ_fc0_46036l2_integ_fc0_46038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l2_integ_fc0_layer_call_and_return_conditional_losses_459902&
$l2_integ_fc0/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall-l2_integ_fc0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_460112
activation_19/PartitionedCallЁ
IdentityIdentity&activation_19/PartitionedCall:output:0%^l2_integ_fc0/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::2L
$l2_integ_fc0/StatefulPartitionedCall$l2_integ_fc0/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџA
 
_user_specified_nameinputs
Ќ
d
H__inference_activation_19_layer_call_and_return_conditional_losses_46144

inputs
identityN
TanhTanhinputs*
T0*'
_output_shapes
:џџџџџџџџџB2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџB:O K
'
_output_shapes
:џџџџџџџџџB
 
_user_specified_nameinputs
у

,__inference_l2_integ_fc0_layer_call_fn_46139

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l2_integ_fc0_layer_call_and_return_conditional_losses_459902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџA
 
_user_specified_nameinputs
ч

.__inference_l4_integration_layer_call_fn_46120

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_l4_integration_layer_call_and_return_conditional_losses_460622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџA
 
_user_specified_nameinputs


.__inference_l4_integration_layer_call_fn_46050
l2_integ_fc0_input
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalll2_integ_fc0_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_l4_integration_layer_call_and_return_conditional_losses_460432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџA
,
_user_specified_namel2_integ_fc0_input
 

Ы
I__inference_l4_integration_layer_call_and_return_conditional_losses_46091

inputs/
+l2_integ_fc0_matmul_readvariableop_resource0
,l2_integ_fc0_biasadd_readvariableop_resource
identityД
"l2_integ_fc0/MatMul/ReadVariableOpReadVariableOp+l2_integ_fc0_matmul_readvariableop_resource*
_output_shapes

:AB*
dtype02$
"l2_integ_fc0/MatMul/ReadVariableOp
l2_integ_fc0/MatMulMatMulinputs*l2_integ_fc0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2
l2_integ_fc0/MatMulГ
#l2_integ_fc0/BiasAdd/ReadVariableOpReadVariableOp,l2_integ_fc0_biasadd_readvariableop_resource*
_output_shapes
:B*
dtype02%
#l2_integ_fc0/BiasAdd/ReadVariableOpЕ
l2_integ_fc0/BiasAddBiasAddl2_integ_fc0/MatMul:product:0+l2_integ_fc0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2
l2_integ_fc0/BiasAdd
activation_19/TanhTanhl2_integ_fc0/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџB2
activation_19/Tanhj
IdentityIdentityactivation_19/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA:::O K
'
_output_shapes
:џџџџџџџџџA
 
_user_specified_nameinputs
Ц
я
__inference__traced_save_46178
file_prefixA
=savev2_l4_integration_l2_integ_fc0_kernel_read_readvariableop?
;savev2_l4_integration_l2_integ_fc0_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_63222f512b6147b28eb0a4f937a15f54/part2	
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ё
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
SaveV2/shape_and_slicesИ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_l4_integration_l2_integ_fc0_kernel_read_readvariableop;savev2_l4_integration_l2_integ_fc0_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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

identity_1Identity_1:output:0*'
_input_shapes
: :AB:B: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:AB: 

_output_shapes
:B:

_output_shapes
: 

I
-__inference_activation_19_layer_call_fn_46149

inputs
identityЦ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_460112
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџB:O K
'
_output_shapes
:џџџџџџџџџB
 
_user_specified_nameinputs
Ќ
Ы
I__inference_l4_integration_layer_call_and_return_conditional_losses_46020
l2_integ_fc0_input
l2_integ_fc0_46001
l2_integ_fc0_46003
identityЂ$l2_integ_fc0/StatefulPartitionedCallБ
$l2_integ_fc0/StatefulPartitionedCallStatefulPartitionedCalll2_integ_fc0_inputl2_integ_fc0_46001l2_integ_fc0_46003*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_l2_integ_fc0_layer_call_and_return_conditional_losses_459902&
$l2_integ_fc0/StatefulPartitionedCall
activation_19/PartitionedCallPartitionedCall-l2_integ_fc0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџB* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_activation_19_layer_call_and_return_conditional_losses_460112
activation_19/PartitionedCallЁ
IdentityIdentity&activation_19/PartitionedCall:output:0%^l2_integ_fc0/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA::2L
$l2_integ_fc0/StatefulPartitionedCall$l2_integ_fc0/StatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџA
,
_user_specified_namel2_integ_fc0_input
Ќ
d
H__inference_activation_19_layer_call_and_return_conditional_losses_46011

inputs
identityN
TanhTanhinputs*
T0*'
_output_shapes
:џџџџџџџџџB2
Tanh\
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџB:O K
'
_output_shapes
:џџџџџџџџџB
 
_user_specified_nameinputs
Р
с
!__inference__traced_restore_46194
file_prefix7
3assignvariableop_l4_integration_l2_integ_fc0_kernel7
3assignvariableop_1_l4_integration_l2_integ_fc0_bias

identity_3ЂAssignVariableOpЂAssignVariableOp_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ё
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
RestoreV2/shape_and_slicesК
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

IdentityВ
AssignVariableOpAssignVariableOp3assignvariableop_l4_integration_l2_integ_fc0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1И
AssignVariableOp_1AssignVariableOp3assignvariableop_1_l4_integration_l2_integ_fc0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_2

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
И
Ь
 __inference__wrapped_model_45976
l2_integ_fc0_input>
:l4_integration_l2_integ_fc0_matmul_readvariableop_resource?
;l4_integration_l2_integ_fc0_biasadd_readvariableop_resource
identityс
1l4_integration/l2_integ_fc0/MatMul/ReadVariableOpReadVariableOp:l4_integration_l2_integ_fc0_matmul_readvariableop_resource*
_output_shapes

:AB*
dtype023
1l4_integration/l2_integ_fc0/MatMul/ReadVariableOpг
"l4_integration/l2_integ_fc0/MatMulMatMull2_integ_fc0_input9l4_integration/l2_integ_fc0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2$
"l4_integration/l2_integ_fc0/MatMulр
2l4_integration/l2_integ_fc0/BiasAdd/ReadVariableOpReadVariableOp;l4_integration_l2_integ_fc0_biasadd_readvariableop_resource*
_output_shapes
:B*
dtype024
2l4_integration/l2_integ_fc0/BiasAdd/ReadVariableOpё
#l4_integration/l2_integ_fc0/BiasAddBiasAdd,l4_integration/l2_integ_fc0/MatMul:product:0:l4_integration/l2_integ_fc0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџB2%
#l4_integration/l2_integ_fc0/BiasAddЎ
!l4_integration/activation_19/TanhTanh,l4_integration/l2_integ_fc0/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџB2#
!l4_integration/activation_19/Tanhy
IdentityIdentity%l4_integration/activation_19/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџB2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџA:::[ W
'
_output_shapes
:џџџџџџџџџA
,
_user_specified_namel2_integ_fc0_input"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ц
serving_defaultВ
Q
l2_integ_fc0_input;
$serving_default_l2_integ_fc0_input:0џџџџџџџџџAA
activation_190
StatefulPartitionedCall:0џџџџџџџџџBtensorflow/serving/predict:Q
№
layer_with_weights-0
layer-0
layer-1
regularization_losses
trainable_variables
	variables
	keras_api

signatures
*#&call_and_return_all_conditional_losses
$__call__
%_default_save_signature"
_tf_keras_sequentialс{"class_name": "Sequential", "name": "l4_integration", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "l4_integration", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 65]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "l2_integ_fc0_input"}}, {"class_name": "Dense", "config": {"name": "l2_integ_fc0", "trainable": true, "dtype": "float32", "units": 66, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "tanh"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 65}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 65]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "l4_integration", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 65]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "l2_integ_fc0_input"}}, {"class_name": "Dense", "config": {"name": "l2_integ_fc0", "trainable": true, "dtype": "float32", "units": 66, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "tanh"}}]}}}

_inbound_nodes

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
*&&call_and_return_all_conditional_losses
'__call__"з
_tf_keras_layerН{"class_name": "Dense", "name": "l2_integ_fc0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "l2_integ_fc0", "trainable": true, "dtype": "float32", "units": 66, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 65}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 65]}}
ы
_inbound_nodes
regularization_losses
trainable_variables
	variables
	keras_api
*(&call_and_return_all_conditional_losses
)__call__"Ш
_tf_keras_layerЎ{"class_name": "Activation", "name": "activation_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_19", "trainable": true, "dtype": "float32", "activation": "tanh"}}
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
Ъ
regularization_losses
metrics

layers
non_trainable_variables
trainable_variables
layer_metrics
layer_regularization_losses
	variables
$__call__
%_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
,
*serving_default"
signature_map
 "
trackable_list_wrapper
4:2AB2"l4_integration/l2_integ_fc0/kernel
.:,B2 l4_integration/l2_integ_fc0/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
­
regularization_losses
layer_metrics
metrics
non_trainable_variables
trainable_variables

layers
layer_regularization_losses
	variables
'__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses
layer_metrics
metrics
 non_trainable_variables
trainable_variables

!layers
"layer_regularization_losses
	variables
)__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
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
ђ2я
I__inference_l4_integration_layer_call_and_return_conditional_losses_46030
I__inference_l4_integration_layer_call_and_return_conditional_losses_46020
I__inference_l4_integration_layer_call_and_return_conditional_losses_46091
I__inference_l4_integration_layer_call_and_return_conditional_losses_46102Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
.__inference_l4_integration_layer_call_fn_46120
.__inference_l4_integration_layer_call_fn_46050
.__inference_l4_integration_layer_call_fn_46111
.__inference_l4_integration_layer_call_fn_46069Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
щ2ц
 __inference__wrapped_model_45976С
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *1Ђ.
,)
l2_integ_fc0_inputџџџџџџџџџA
ё2ю
G__inference_l2_integ_fc0_layer_call_and_return_conditional_losses_46130Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_l2_integ_fc0_layer_call_fn_46139Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_activation_19_layer_call_and_return_conditional_losses_46144Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_activation_19_layer_call_fn_46149Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
=B;
#__inference_signature_wrapper_46080l2_integ_fc0_inputЅ
 __inference__wrapped_model_45976	
;Ђ8
1Ђ.
,)
l2_integ_fc0_inputџџџџџџџџџA
Њ "=Њ:
8
activation_19'$
activation_19џџџџџџџџџBЄ
H__inference_activation_19_layer_call_and_return_conditional_losses_46144X/Ђ,
%Ђ"
 
inputsџџџџџџџџџB
Њ "%Ђ"

0џџџџџџџџџB
 |
-__inference_activation_19_layer_call_fn_46149K/Ђ,
%Ђ"
 
inputsџџџџџџџџџB
Њ "џџџџџџџџџBЇ
G__inference_l2_integ_fc0_layer_call_and_return_conditional_losses_46130\	
/Ђ,
%Ђ"
 
inputsџџџџџџџџџA
Њ "%Ђ"

0џџџџџџџџџB
 
,__inference_l2_integ_fc0_layer_call_fn_46139O	
/Ђ,
%Ђ"
 
inputsџџџџџџџџџA
Њ "џџџџџџџџџBН
I__inference_l4_integration_layer_call_and_return_conditional_losses_46020p	
CЂ@
9Ђ6
,)
l2_integ_fc0_inputџџџџџџџџџA
p

 
Њ "%Ђ"

0џџџџџџџџџB
 Н
I__inference_l4_integration_layer_call_and_return_conditional_losses_46030p	
CЂ@
9Ђ6
,)
l2_integ_fc0_inputџџџџџџџџџA
p 

 
Њ "%Ђ"

0џџџџџџџџџB
 Б
I__inference_l4_integration_layer_call_and_return_conditional_losses_46091d	
7Ђ4
-Ђ*
 
inputsџџџџџџџџџA
p

 
Њ "%Ђ"

0џџџџџџџџџB
 Б
I__inference_l4_integration_layer_call_and_return_conditional_losses_46102d	
7Ђ4
-Ђ*
 
inputsџџџџџџџџџA
p 

 
Њ "%Ђ"

0џџџџџџџџџB
 
.__inference_l4_integration_layer_call_fn_46050c	
CЂ@
9Ђ6
,)
l2_integ_fc0_inputџџџџџџџџџA
p

 
Њ "џџџџџџџџџB
.__inference_l4_integration_layer_call_fn_46069c	
CЂ@
9Ђ6
,)
l2_integ_fc0_inputџџџџџџџџџA
p 

 
Њ "џџџџџџџџџB
.__inference_l4_integration_layer_call_fn_46111W	
7Ђ4
-Ђ*
 
inputsџџџџџџџџџA
p

 
Њ "џџџџџџџџџB
.__inference_l4_integration_layer_call_fn_46120W	
7Ђ4
-Ђ*
 
inputsџџџџџџџџџA
p 

 
Њ "џџџџџџџџџBО
#__inference_signature_wrapper_46080	
QЂN
Ђ 
GЊD
B
l2_integ_fc0_input,)
l2_integ_fc0_inputџџџџџџџџџA"=Њ:
8
activation_19'$
activation_19џџџџџџџџџB