??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58??
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
?
Adam/v/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/v/dense_17/bias
y
(Adam/v/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/bias*
_output_shapes
:*
dtype0
?
Adam/m/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/m/dense_17/bias
y
(Adam/m/dense_17/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/bias*
_output_shapes
:*
dtype0
?
Adam/v/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/v/dense_17/kernel
?
*Adam/v/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_17/kernel*
_output_shapes
:	?*
dtype0
?
Adam/m/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/m/dense_17/kernel
?
*Adam/m/dense_17/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_17/kernel*
_output_shapes
:	?*
dtype0
?
Adam/v/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/v/dense_16/bias
z
(Adam/v/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/bias*
_output_shapes	
:?*
dtype0
?
Adam/m/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/m/dense_16/bias
z
(Adam/m/dense_16/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/bias*
_output_shapes	
:?*
dtype0
?
Adam/v/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/v/dense_16/kernel
?
*Adam/v/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_16/kernel* 
_output_shapes
:
??*
dtype0
?
Adam/m/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/m/dense_16/kernel
?
*Adam/m/dense_16/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_16/kernel* 
_output_shapes
:
??*
dtype0
?
!Adam/v/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/v/batch_normalization_8/beta
?
5Adam/v/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp!Adam/v/batch_normalization_8/beta*
_output_shapes
: *
dtype0
?
!Adam/m/batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/m/batch_normalization_8/beta
?
5Adam/m/batch_normalization_8/beta/Read/ReadVariableOpReadVariableOp!Adam/m/batch_normalization_8/beta*
_output_shapes
: *
dtype0
?
"Adam/v/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/v/batch_normalization_8/gamma
?
6Adam/v/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_8/gamma*
_output_shapes
: *
dtype0
?
"Adam/m/batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/m/batch_normalization_8/gamma
?
6Adam/m/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_8/gamma*
_output_shapes
: *
dtype0
?
Adam/v/conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/conv2d_35/bias
{
)Adam/v/conv2d_35/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_35/bias*
_output_shapes
: *
dtype0
?
Adam/m/conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/conv2d_35/bias
{
)Adam/m/conv2d_35/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_35/bias*
_output_shapes
: *
dtype0
?
Adam/v/conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/v/conv2d_35/kernel
?
+Adam/v/conv2d_35/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_35/kernel*&
_output_shapes
:  *
dtype0
?
Adam/m/conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/m/conv2d_35/kernel
?
+Adam/m/conv2d_35/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_35/kernel*&
_output_shapes
:  *
dtype0
?
Adam/v/conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/conv2d_34/bias
{
)Adam/v/conv2d_34/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_34/bias*
_output_shapes
: *
dtype0
?
Adam/m/conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/conv2d_34/bias
{
)Adam/m/conv2d_34/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_34/bias*
_output_shapes
: *
dtype0
?
Adam/v/conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/v/conv2d_34/kernel
?
+Adam/v/conv2d_34/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_34/kernel*&
_output_shapes
:  *
dtype0
?
Adam/m/conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/m/conv2d_34/kernel
?
+Adam/m/conv2d_34/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_34/kernel*&
_output_shapes
:  *
dtype0
?
Adam/v/conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/conv2d_33/bias
{
)Adam/v/conv2d_33/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_33/bias*
_output_shapes
: *
dtype0
?
Adam/m/conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/conv2d_33/bias
{
)Adam/m/conv2d_33/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_33/bias*
_output_shapes
: *
dtype0
?
Adam/v/conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/v/conv2d_33/kernel
?
+Adam/v/conv2d_33/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_33/kernel*&
_output_shapes
:  *
dtype0
?
Adam/m/conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/m/conv2d_33/kernel
?
+Adam/m/conv2d_33/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_33/kernel*&
_output_shapes
:  *
dtype0
?
Adam/v/conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/v/conv2d_32/bias
{
)Adam/v/conv2d_32/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_32/bias*
_output_shapes
: *
dtype0
?
Adam/m/conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/m/conv2d_32/bias
{
)Adam/m/conv2d_32/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_32/bias*
_output_shapes
: *
dtype0
?
Adam/v/conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/v/conv2d_32/kernel
?
+Adam/v/conv2d_32/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_32/kernel*&
_output_shapes
: *
dtype0
?
Adam/m/conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/m/conv2d_32/kernel
?
+Adam/m/conv2d_32/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_32/kernel*&
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
l
save_counterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namesave_counter
e
 save_counter/Read/ReadVariableOpReadVariableOpsave_counter*
_output_shapes
: *
dtype0	
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:*
dtype0
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	?*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:?*
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
??*
dtype0
?
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_8/moving_variance
?
9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes
: *
dtype0
?
!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_8/moving_mean
?
5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_8/beta
?
.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes
: *
dtype0
?
batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_8/gamma
?
/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes
: *
dtype0
t
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_35/bias
m
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes
: *
dtype0
?
conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_35/kernel
}
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
: *
dtype0
?
conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
: *
dtype0
?
conv2d_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_32/bias
m
"conv2d_32/bias/Read/ReadVariableOpReadVariableOpconv2d_32/bias*
_output_shapes
: *
dtype0
?
conv2d_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_32/kernel
}
$conv2d_32/kernel/Read/ReadVariableOpReadVariableOpconv2d_32/kernel*&
_output_shapes
: *
dtype0
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_varianceconv2d_35/kernelconv2d_35/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_313599

NoOpNoOp
?g
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?f
value?fB?f B?f
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2
	
conv3
	conv4
	pool1
	pool4
	batchnorm
	dropout40
flatten
d128
d7

checkpoint
	optimizer

signatures*
z
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15*
j
0
1
2
3
4
5
6
7
8
9
"10
#11
$12
%13*
* 
?
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
+trace_0
,trace_1
-trace_2
.trace_3* 
6
/trace_0
0trace_1
1trace_2
2trace_3* 
* 
?
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

kernel
bias
 9_jit_compiled_convolution_op*
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

kernel
bias
 @_jit_compiled_convolution_op*
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

kernel
bias
 G_jit_compiled_convolution_op*
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

kernel
bias
 N_jit_compiled_convolution_op*
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses* 
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
?
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
aaxis
	gamma
beta
 moving_mean
!moving_variance*
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator* 
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses* 
?
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

"kernel
#bias*
?
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

$kernel
%bias*

model
{save_counter*
?
|
_variables
}_iterations
~_learning_rate
_index_dict
?
_momentums
?_velocities
?_update_step_xla*

?serving_default* 
PJ
VARIABLE_VALUEconv2d_32/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_32/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_33/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_33/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_34/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_34/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_35/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_35/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_8/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_8/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_8/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_8/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_16/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_16/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEdense_17/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEdense_17/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*
R
0
	1

2
3
4
5
6
7
8
9
10*

?0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
0
1
 2
!3*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

"0
#1*

"0
#1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 

$0
%1*

$0
%1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
XR
VARIABLE_VALUEsave_counter2checkpoint/save_counter/.ATTRIBUTES/VARIABLE_VALUE*
?
}0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
x
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13*
x
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13*
?
?trace_0
?trace_1
?trace_2
?trace_3
?trace_4
?trace_5
?trace_6
?trace_7
?trace_8
?trace_9
?trace_10
?trace_11
?trace_12
?trace_13* 
* 
<
?	variables
?	keras_api

?total

?count*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

 0
!1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
b\
VARIABLE_VALUEAdam/m/conv2d_32/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_32/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_32/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_32/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_33/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_33/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_33/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_33/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_34/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_34/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_34/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_34/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_35/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_35/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_35/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_35/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_8/gamma2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_8/gamma2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/m/batch_normalization_8/beta2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE!Adam/v/batch_normalization_8/beta2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_16/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_16/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_16/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_16/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_17/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_17/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_17/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_17/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_32/kernel/Read/ReadVariableOp"conv2d_32/bias/Read/ReadVariableOp$conv2d_33/kernel/Read/ReadVariableOp"conv2d_33/bias/Read/ReadVariableOp$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOp$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOp save_counter/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp+Adam/m/conv2d_32/kernel/Read/ReadVariableOp+Adam/v/conv2d_32/kernel/Read/ReadVariableOp)Adam/m/conv2d_32/bias/Read/ReadVariableOp)Adam/v/conv2d_32/bias/Read/ReadVariableOp+Adam/m/conv2d_33/kernel/Read/ReadVariableOp+Adam/v/conv2d_33/kernel/Read/ReadVariableOp)Adam/m/conv2d_33/bias/Read/ReadVariableOp)Adam/v/conv2d_33/bias/Read/ReadVariableOp+Adam/m/conv2d_34/kernel/Read/ReadVariableOp+Adam/v/conv2d_34/kernel/Read/ReadVariableOp)Adam/m/conv2d_34/bias/Read/ReadVariableOp)Adam/v/conv2d_34/bias/Read/ReadVariableOp+Adam/m/conv2d_35/kernel/Read/ReadVariableOp+Adam/v/conv2d_35/kernel/Read/ReadVariableOp)Adam/m/conv2d_35/bias/Read/ReadVariableOp)Adam/v/conv2d_35/bias/Read/ReadVariableOp6Adam/m/batch_normalization_8/gamma/Read/ReadVariableOp6Adam/v/batch_normalization_8/gamma/Read/ReadVariableOp5Adam/m/batch_normalization_8/beta/Read/ReadVariableOp5Adam/v/batch_normalization_8/beta/Read/ReadVariableOp*Adam/m/dense_16/kernel/Read/ReadVariableOp*Adam/v/dense_16/kernel/Read/ReadVariableOp(Adam/m/dense_16/bias/Read/ReadVariableOp(Adam/v/dense_16/bias/Read/ReadVariableOp*Adam/m/dense_17/kernel/Read/ReadVariableOp*Adam/v/dense_17/kernel/Read/ReadVariableOp(Adam/m/dense_17/bias/Read/ReadVariableOp(Adam/v/dense_17/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*>
Tin7
523		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_314294
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_32/kernelconv2d_32/biasconv2d_33/kernelconv2d_33/biasconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense_16/kerneldense_16/biasdense_17/kerneldense_17/biassave_counter	iterationlearning_rateAdam/m/conv2d_32/kernelAdam/v/conv2d_32/kernelAdam/m/conv2d_32/biasAdam/v/conv2d_32/biasAdam/m/conv2d_33/kernelAdam/v/conv2d_33/kernelAdam/m/conv2d_33/biasAdam/v/conv2d_33/biasAdam/m/conv2d_34/kernelAdam/v/conv2d_34/kernelAdam/m/conv2d_34/biasAdam/v/conv2d_34/biasAdam/m/conv2d_35/kernelAdam/v/conv2d_35/kernelAdam/m/conv2d_35/biasAdam/v/conv2d_35/bias"Adam/m/batch_normalization_8/gamma"Adam/v/batch_normalization_8/gamma!Adam/m/batch_normalization_8/beta!Adam/v/batch_normalization_8/betaAdam/m/dense_16/kernelAdam/v/dense_16/kernelAdam/m/dense_16/biasAdam/v/dense_16/biasAdam/m/dense_17/kernelAdam/v/dense_17/kernelAdam/m/dense_17/biasAdam/v/dense_17/biastotalcount*=
Tin6
422*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_314451??	
?
Q
#__inference__update_step_xla_313870
gradient
variable:
??*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
??: *
	_noinline(:J F
 
_output_shapes
:
??
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_314085

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_34_layer_call_fn_313934

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313099w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????66 : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????66 
 
_user_specified_nameinputs
?
L
#__inference__update_step_xla_313875
gradient
variable:	?*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:?: *
	_noinline(:E A

_output_shapes	
:?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313125

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_313985

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
W
#__inference__update_step_xla_313850
gradient"
variable:  *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:  : *
	_noinline(:P L
&
_output_shapes
:  
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
??
?
"__inference__traced_restore_314451
file_prefix;
!assignvariableop_conv2d_32_kernel: /
!assignvariableop_1_conv2d_32_bias: =
#assignvariableop_2_conv2d_33_kernel:  /
!assignvariableop_3_conv2d_33_bias: =
#assignvariableop_4_conv2d_34_kernel:  /
!assignvariableop_5_conv2d_34_bias: =
#assignvariableop_6_conv2d_35_kernel:  /
!assignvariableop_7_conv2d_35_bias: <
.assignvariableop_8_batch_normalization_8_gamma: ;
-assignvariableop_9_batch_normalization_8_beta: C
5assignvariableop_10_batch_normalization_8_moving_mean: G
9assignvariableop_11_batch_normalization_8_moving_variance: 7
#assignvariableop_12_dense_16_kernel:
??0
!assignvariableop_13_dense_16_bias:	?6
#assignvariableop_14_dense_17_kernel:	?/
!assignvariableop_15_dense_17_bias:*
 assignvariableop_16_save_counter:	 '
assignvariableop_17_iteration:	 +
!assignvariableop_18_learning_rate: E
+assignvariableop_19_adam_m_conv2d_32_kernel: E
+assignvariableop_20_adam_v_conv2d_32_kernel: 7
)assignvariableop_21_adam_m_conv2d_32_bias: 7
)assignvariableop_22_adam_v_conv2d_32_bias: E
+assignvariableop_23_adam_m_conv2d_33_kernel:  E
+assignvariableop_24_adam_v_conv2d_33_kernel:  7
)assignvariableop_25_adam_m_conv2d_33_bias: 7
)assignvariableop_26_adam_v_conv2d_33_bias: E
+assignvariableop_27_adam_m_conv2d_34_kernel:  E
+assignvariableop_28_adam_v_conv2d_34_kernel:  7
)assignvariableop_29_adam_m_conv2d_34_bias: 7
)assignvariableop_30_adam_v_conv2d_34_bias: E
+assignvariableop_31_adam_m_conv2d_35_kernel:  E
+assignvariableop_32_adam_v_conv2d_35_kernel:  7
)assignvariableop_33_adam_m_conv2d_35_bias: 7
)assignvariableop_34_adam_v_conv2d_35_bias: D
6assignvariableop_35_adam_m_batch_normalization_8_gamma: D
6assignvariableop_36_adam_v_batch_normalization_8_gamma: C
5assignvariableop_37_adam_m_batch_normalization_8_beta: C
5assignvariableop_38_adam_v_batch_normalization_8_beta: >
*assignvariableop_39_adam_m_dense_16_kernel:
??>
*assignvariableop_40_adam_v_dense_16_kernel:
??7
(assignvariableop_41_adam_m_dense_16_bias:	?7
(assignvariableop_42_adam_v_dense_16_bias:	?=
*assignvariableop_43_adam_m_dense_17_kernel:	?=
*assignvariableop_44_adam_v_dense_17_kernel:	?6
(assignvariableop_45_adam_m_dense_17_bias:6
(assignvariableop_46_adam_v_dense_17_bias:#
assignvariableop_47_total: #
assignvariableop_48_count: 
identity_50??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB2checkpoint/save_counter/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_32_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_32_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_33_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_33_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_34_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_34_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_35_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_35_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_8_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_8_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_8_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_8_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_16_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_16_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_17_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_17_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp assignvariableop_16_save_counterIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_iterationIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_learning_rateIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_m_conv2d_32_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp+assignvariableop_20_adam_v_conv2d_32_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_m_conv2d_32_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_v_conv2d_32_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_m_conv2d_33_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_v_conv2d_33_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_m_conv2d_33_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_v_conv2d_33_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_m_conv2d_34_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_v_conv2d_34_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_m_conv2d_34_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_v_conv2d_34_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_m_conv2d_35_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_v_conv2d_35_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_m_conv2d_35_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_v_conv2d_35_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp6assignvariableop_35_adam_m_batch_normalization_8_gammaIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp6assignvariableop_36_adam_v_batch_normalization_8_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_m_batch_normalization_8_betaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_v_batch_normalization_8_betaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_m_dense_16_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_v_dense_16_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp(assignvariableop_41_adam_m_dense_16_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_v_dense_16_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_m_dense_17_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_v_dense_17_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp(assignvariableop_45_adam_m_dense_17_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_v_dense_17_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ?	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
M
1__inference_max_pooling2d_16_layer_call_fn_313970

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_312978

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313905

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_dense_16_layer_call_and_return_conditional_losses_313151

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_313975

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313965

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_dense_17_layer_call_fn_314114

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_313167o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?8
?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313558
input_1*
conv2d_32_313513: 
conv2d_32_313515: *
conv2d_33_313519:  
conv2d_33_313521: *
conv2d_34_313525:  
conv2d_34_313527: *
batch_normalization_8_313530: *
batch_normalization_8_313532: *
batch_normalization_8_313534: *
batch_normalization_8_313536: *
conv2d_35_313540:  
conv2d_35_313542: #
dense_16_313547:
??
dense_16_313549:	?"
dense_17_313552:	?
dense_17_313554:
identity??-batch_normalization_8/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_32_313513conv2d_32_313515*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313063?
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????oo * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_33_313519conv2d_33_313521*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313081?
"max_pooling2d_16/PartitionedCall_1PartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_16/PartitionedCall_1:output:0conv2d_34_313525conv2d_34_313527*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313099?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_8_313530batch_normalization_8_313532batch_normalization_8_313534batch_normalization_8_313536*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_313034?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_313355?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_35_313540conv2d_35_313542*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313125?
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_312978?
flatten_8/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_313138?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_16_313547dense_16_313549*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_313151?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_313552dense_17_313554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_313167x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^batch_normalization_8/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_313167

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_314029

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313945

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????66 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????66 
 
_user_specified_nameinputs
?N
?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313737

inputsB
(conv2d_32_conv2d_readvariableop_resource: 7
)conv2d_32_biasadd_readvariableop_resource: B
(conv2d_33_conv2d_readvariableop_resource:  7
)conv2d_33_biasadd_readvariableop_resource: B
(conv2d_34_conv2d_readvariableop_resource:  7
)conv2d_34_biasadd_readvariableop_resource: ;
-batch_normalization_8_readvariableop_resource: =
/batch_normalization_8_readvariableop_1_resource: L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_35_conv2d_readvariableop_resource:  7
)conv2d_35_biasadd_readvariableop_resource: ;
'dense_16_matmul_readvariableop_resource:
??7
(dense_16_biasadd_readvariableop_resource:	?:
'dense_17_matmul_readvariableop_resource:	?6
(dense_17_biasadd_readvariableop_resource:
identity??5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1? conv2d_32/BiasAdd/ReadVariableOp?conv2d_32/Conv2D/ReadVariableOp? conv2d_33/BiasAdd/ReadVariableOp?conv2d_33/Conv2D/ReadVariableOp? conv2d_34/BiasAdd/ReadVariableOp?conv2d_34/Conv2D/ReadVariableOp? conv2d_35/BiasAdd/ReadVariableOp?conv2d_35/Conv2D/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_32/Conv2DConv2Dinputs'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? n
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? ?
max_pooling2d_16/MaxPoolMaxPoolconv2d_32/Relu:activations:0*/
_output_shapes
:?????????oo *
ksize
*
paddingVALID*
strides
?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_33/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm *
paddingVALID*
strides
?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm l
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:?????????mm ?
max_pooling2d_16/MaxPool_1MaxPoolconv2d_33/Relu:activations:0*/
_output_shapes
:?????????66 *
ksize
*
paddingVALID*
strides
?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_34/Conv2DConv2D#max_pooling2d_16/MaxPool_1:output:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_34/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_35/Conv2DConv2D*batch_normalization_8/FusedBatchNormV3:y:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
max_pooling2d_17/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_8/ReshapeReshape!max_pooling2d_17/MaxPool:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:???????????
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_16/MatMulMatMulflatten_8/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp6^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313925

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????mm i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????mm w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????oo : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????oo 
 
_user_specified_nameinputs
?^
?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313809

inputsB
(conv2d_32_conv2d_readvariableop_resource: 7
)conv2d_32_biasadd_readvariableop_resource: B
(conv2d_33_conv2d_readvariableop_resource:  7
)conv2d_33_biasadd_readvariableop_resource: B
(conv2d_34_conv2d_readvariableop_resource:  7
)conv2d_34_biasadd_readvariableop_resource: ;
-batch_normalization_8_readvariableop_resource: =
/batch_normalization_8_readvariableop_1_resource: L
>batch_normalization_8_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_35_conv2d_readvariableop_resource:  7
)conv2d_35_biasadd_readvariableop_resource: ;
'dense_16_matmul_readvariableop_resource:
??7
(dense_16_biasadd_readvariableop_resource:	?:
'dense_17_matmul_readvariableop_resource:	?6
(dense_17_biasadd_readvariableop_resource:
identity??$batch_normalization_8/AssignNewValue?&batch_normalization_8/AssignNewValue_1?5batch_normalization_8/FusedBatchNormV3/ReadVariableOp?7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_8/ReadVariableOp?&batch_normalization_8/ReadVariableOp_1? conv2d_32/BiasAdd/ReadVariableOp?conv2d_32/Conv2D/ReadVariableOp? conv2d_33/BiasAdd/ReadVariableOp?conv2d_33/Conv2D/ReadVariableOp? conv2d_34/BiasAdd/ReadVariableOp?conv2d_34/Conv2D/ReadVariableOp? conv2d_35/BiasAdd/ReadVariableOp?conv2d_35/Conv2D/ReadVariableOp?dense_16/BiasAdd/ReadVariableOp?dense_16/MatMul/ReadVariableOp?dense_17/BiasAdd/ReadVariableOp?dense_17/MatMul/ReadVariableOp?
conv2d_32/Conv2D/ReadVariableOpReadVariableOp(conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_32/Conv2DConv2Dinputs'conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
?
 conv2d_32/BiasAdd/ReadVariableOpReadVariableOp)conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_32/BiasAddBiasAddconv2d_32/Conv2D:output:0(conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? n
conv2d_32/ReluReluconv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? ?
max_pooling2d_16/MaxPoolMaxPoolconv2d_32/Relu:activations:0*/
_output_shapes
:?????????oo *
ksize
*
paddingVALID*
strides
?
conv2d_33/Conv2D/ReadVariableOpReadVariableOp(conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_33/Conv2DConv2D!max_pooling2d_16/MaxPool:output:0'conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm *
paddingVALID*
strides
?
 conv2d_33/BiasAdd/ReadVariableOpReadVariableOp)conv2d_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_33/BiasAddBiasAddconv2d_33/Conv2D:output:0(conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm l
conv2d_33/ReluReluconv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:?????????mm ?
max_pooling2d_16/MaxPool_1MaxPoolconv2d_33/Relu:activations:0*/
_output_shapes
:?????????66 *
ksize
*
paddingVALID*
strides
?
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_34/Conv2DConv2D#max_pooling2d_16/MaxPool_1:output:0'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
$batch_normalization_8/ReadVariableOpReadVariableOp-batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype0?
&batch_normalization_8/ReadVariableOp_1ReadVariableOp/batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype0?
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_34/Relu:activations:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_8/AssignNewValueAssignVariableOp>batch_normalization_8_fusedbatchnormv3_readvariableop_resource3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
&batch_normalization_8/AssignNewValue_1AssignVariableOp@batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(\
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU???
dropout_8/dropout/MulMul*batch_normalization_8/FusedBatchNormV3:y:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:????????? q
dropout_8/dropout/ShapeShape*batch_normalization_8/FusedBatchNormV3:y:0*
T0*
_output_shapes
:?
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0e
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? ^
dropout_8/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
dropout_8/dropout/SelectV2SelectV2"dropout_8/dropout/GreaterEqual:z:0dropout_8/dropout/Mul:z:0"dropout_8/dropout/Const_1:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
conv2d_35/Conv2DConv2D#dropout_8/dropout/SelectV2:output:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? l
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
max_pooling2d_17/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
`
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_8/ReshapeReshape!max_pooling2d_17/MaxPool:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:???????????
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_16/MatMulMatMulflatten_8/Reshape:output:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
IdentityIdentitydense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_16^batch_normalization_8/FusedBatchNormV3/ReadVariableOp8^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_8/ReadVariableOp'^batch_normalization_8/ReadVariableOp_1!^conv2d_32/BiasAdd/ReadVariableOp ^conv2d_32/Conv2D/ReadVariableOp!^conv2d_33/BiasAdd/ReadVariableOp ^conv2d_33/Conv2D/ReadVariableOp!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_12n
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp5batch_normalization_8/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_17batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_8/ReadVariableOp$batch_normalization_8/ReadVariableOp2P
&batch_normalization_8/ReadVariableOp_1&batch_normalization_8/ReadVariableOp_12D
 conv2d_32/BiasAdd/ReadVariableOp conv2d_32/BiasAdd/ReadVariableOp2B
conv2d_32/Conv2D/ReadVariableOpconv2d_32/Conv2D/ReadVariableOp2D
 conv2d_33/BiasAdd/ReadVariableOp conv2d_33/BiasAdd/ReadVariableOp2B
conv2d_33/Conv2D/ReadVariableOpconv2d_33/Conv2D/ReadVariableOp2D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_313003

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
&__inference_cnn_8_layer_call_fn_313673

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_8_layer_call_and_return_conditional_losses_313375o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?4
?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313510
input_1*
conv2d_32_313466: 
conv2d_32_313468: *
conv2d_33_313472:  
conv2d_33_313474: *
conv2d_34_313478:  
conv2d_34_313480: *
batch_normalization_8_313483: *
batch_normalization_8_313485: *
batch_normalization_8_313487: *
batch_normalization_8_313489: *
conv2d_35_313492:  
conv2d_35_313494: #
dense_16_313499:
??
dense_16_313501:	?"
dense_17_313504:	?
dense_17_313506:
identity??-batch_normalization_8/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_32_313466conv2d_32_313468*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313063?
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????oo * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_33_313472conv2d_33_313474*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313081?
"max_pooling2d_16/PartitionedCall_1PartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_16/PartitionedCall_1:output:0conv2d_34_313478conv2d_34_313480*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313099?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_8_313483batch_normalization_8_313485batch_normalization_8_313487batch_normalization_8_313489*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_313003?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_35_313492conv2d_35_313494*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313125?
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_312978?
flatten_8/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_313138?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_16_313499dense_16_313501*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_313151?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_313504dense_17_313506*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_313167x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^batch_normalization_8/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
h
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_dense_16_layer_call_fn_314094

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_313151p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
W
#__inference__update_step_xla_313840
gradient"
variable:  *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:  : *
	_noinline(:P L
&
_output_shapes
:  
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
c
*__inference_dropout_8_layer_call_fn_314057

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_313355w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
&__inference_cnn_8_layer_call_fn_313463
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_8_layer_call_and_return_conditional_losses_313375o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?8
?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313375

inputs*
conv2d_32_313317: 
conv2d_32_313319: *
conv2d_33_313323:  
conv2d_33_313325: *
conv2d_34_313329:  
conv2d_34_313331: *
batch_normalization_8_313334: *
batch_normalization_8_313336: *
batch_normalization_8_313338: *
batch_normalization_8_313340: *
conv2d_35_313357:  
conv2d_35_313359: #
dense_16_313364:
??
dense_16_313366:	?"
dense_17_313369:	?
dense_17_313371:
identity??-batch_normalization_8/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?!dropout_8/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_32_313317conv2d_32_313319*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313063?
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????oo * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_33_313323conv2d_33_313325*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313081?
"max_pooling2d_16/PartitionedCall_1PartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_16/PartitionedCall_1:output:0conv2d_34_313329conv2d_34_313331*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313099?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_8_313334batch_normalization_8_313336batch_normalization_8_313338batch_normalization_8_313340*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_313034?
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_313355?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0conv2d_35_313357conv2d_35_313359*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313125?
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_312978?
flatten_8/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_313138?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_16_313364dense_16_313366*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_313151?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_313369dense_17_313371*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_313167x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^batch_normalization_8/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?^
?
__inference__traced_save_314294
file_prefix/
+savev2_conv2d_32_kernel_read_readvariableop-
)savev2_conv2d_32_bias_read_readvariableop/
+savev2_conv2d_33_kernel_read_readvariableop-
)savev2_conv2d_33_bias_read_readvariableop/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop+
'savev2_save_counter_read_readvariableop	(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop6
2savev2_adam_m_conv2d_32_kernel_read_readvariableop6
2savev2_adam_v_conv2d_32_kernel_read_readvariableop4
0savev2_adam_m_conv2d_32_bias_read_readvariableop4
0savev2_adam_v_conv2d_32_bias_read_readvariableop6
2savev2_adam_m_conv2d_33_kernel_read_readvariableop6
2savev2_adam_v_conv2d_33_kernel_read_readvariableop4
0savev2_adam_m_conv2d_33_bias_read_readvariableop4
0savev2_adam_v_conv2d_33_bias_read_readvariableop6
2savev2_adam_m_conv2d_34_kernel_read_readvariableop6
2savev2_adam_v_conv2d_34_kernel_read_readvariableop4
0savev2_adam_m_conv2d_34_bias_read_readvariableop4
0savev2_adam_v_conv2d_34_bias_read_readvariableop6
2savev2_adam_m_conv2d_35_kernel_read_readvariableop6
2savev2_adam_v_conv2d_35_kernel_read_readvariableop4
0savev2_adam_m_conv2d_35_bias_read_readvariableop4
0savev2_adam_v_conv2d_35_bias_read_readvariableopA
=savev2_adam_m_batch_normalization_8_gamma_read_readvariableopA
=savev2_adam_v_batch_normalization_8_gamma_read_readvariableop@
<savev2_adam_m_batch_normalization_8_beta_read_readvariableop@
<savev2_adam_v_batch_normalization_8_beta_read_readvariableop5
1savev2_adam_m_dense_16_kernel_read_readvariableop5
1savev2_adam_v_dense_16_kernel_read_readvariableop3
/savev2_adam_m_dense_16_bias_read_readvariableop3
/savev2_adam_v_dense_16_bias_read_readvariableop5
1savev2_adam_m_dense_17_kernel_read_readvariableop5
1savev2_adam_v_dense_17_kernel_read_readvariableop3
/savev2_adam_m_dense_17_bias_read_readvariableop3
/savev2_adam_v_dense_17_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*?
value?B?2B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB2checkpoint/save_counter/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_32_kernel_read_readvariableop)savev2_conv2d_32_bias_read_readvariableop+savev2_conv2d_33_kernel_read_readvariableop)savev2_conv2d_33_bias_read_readvariableop+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop'savev2_save_counter_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop2savev2_adam_m_conv2d_32_kernel_read_readvariableop2savev2_adam_v_conv2d_32_kernel_read_readvariableop0savev2_adam_m_conv2d_32_bias_read_readvariableop0savev2_adam_v_conv2d_32_bias_read_readvariableop2savev2_adam_m_conv2d_33_kernel_read_readvariableop2savev2_adam_v_conv2d_33_kernel_read_readvariableop0savev2_adam_m_conv2d_33_bias_read_readvariableop0savev2_adam_v_conv2d_33_bias_read_readvariableop2savev2_adam_m_conv2d_34_kernel_read_readvariableop2savev2_adam_v_conv2d_34_kernel_read_readvariableop0savev2_adam_m_conv2d_34_bias_read_readvariableop0savev2_adam_v_conv2d_34_bias_read_readvariableop2savev2_adam_m_conv2d_35_kernel_read_readvariableop2savev2_adam_v_conv2d_35_kernel_read_readvariableop0savev2_adam_m_conv2d_35_bias_read_readvariableop0savev2_adam_v_conv2d_35_bias_read_readvariableop=savev2_adam_m_batch_normalization_8_gamma_read_readvariableop=savev2_adam_v_batch_normalization_8_gamma_read_readvariableop<savev2_adam_m_batch_normalization_8_beta_read_readvariableop<savev2_adam_v_batch_normalization_8_beta_read_readvariableop1savev2_adam_m_dense_16_kernel_read_readvariableop1savev2_adam_v_dense_16_kernel_read_readvariableop/savev2_adam_m_dense_16_bias_read_readvariableop/savev2_adam_v_dense_16_bias_read_readvariableop1savev2_adam_m_dense_17_kernel_read_readvariableop1savev2_adam_v_dense_17_kernel_read_readvariableop/savev2_adam_m_dense_17_bias_read_readvariableop/savev2_adam_v_dense_17_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *@
dtypes6
422		?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :  : :  : :  : : : : : :
??:?:	?:: : : : : : : :  :  : : :  :  : : :  :  : : : : : : :
??:
??:?:?:	?:	?::: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: :, (
&
_output_shapes
:  :,!(
&
_output_shapes
:  : "

_output_shapes
: : #

_output_shapes
: : $

_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: :&("
 
_output_shapes
:
??:&)"
 
_output_shapes
:
??:!*

_output_shapes	
:?:!+

_output_shapes	
:?:%,!

_output_shapes
:	?:%-!

_output_shapes
:	?: .

_output_shapes
:: /

_output_shapes
::0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: 
?
?
&__inference_cnn_8_layer_call_fn_313636

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_8_layer_call_and_return_conditional_losses_313174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313081

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????mm i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????mm w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????oo : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????oo 
 
_user_specified_nameinputs
?
K
#__inference__update_step_xla_313835
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
W
#__inference__update_step_xla_313830
gradient"
variable:  *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
:  : *
	_noinline(:P L
&
_output_shapes
:  
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
K
#__inference__update_step_xla_313825
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
W
#__inference__update_step_xla_313820
gradient"
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*'
_input_shapes
: : *
	_noinline(:P L
&
_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
K
#__inference__update_step_xla_313865
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?	
?
D__inference_dense_17_layer_call_and_return_conditional_losses_314124

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
#__inference__update_step_xla_313855
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_313034

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
F
*__inference_flatten_8_layer_call_fn_314079

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_313138a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_35_layer_call_fn_313954

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313125w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_313418

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2d_32_layer_call_fn_313894

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313063y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_dense_16_layer_call_and_return_conditional_losses_314105

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_17_layer_call_fn_313980

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_312978?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?4
?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313174

inputs*
conv2d_32_313064: 
conv2d_32_313066: *
conv2d_33_313082:  
conv2d_33_313084: *
conv2d_34_313100:  
conv2d_34_313102: *
batch_normalization_8_313105: *
batch_normalization_8_313107: *
batch_normalization_8_313109: *
batch_normalization_8_313111: *
conv2d_35_313126:  
conv2d_35_313128: #
dense_16_313152:
??
dense_16_313154:	?"
dense_17_313168:	?
dense_17_313170:
identity??-batch_normalization_8/StatefulPartitionedCall?!conv2d_32/StatefulPartitionedCall?!conv2d_33/StatefulPartitionedCall?!conv2d_34/StatefulPartitionedCall?!conv2d_35/StatefulPartitionedCall? dense_16/StatefulPartitionedCall? dense_17/StatefulPartitionedCall?
!conv2d_32/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_32_313064conv2d_32_313066*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313063?
 max_pooling2d_16/PartitionedCallPartitionedCall*conv2d_32/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????oo * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966?
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_16/PartitionedCall:output:0conv2d_33_313082conv2d_33_313084*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313081?
"max_pooling2d_16/PartitionedCall_1PartitionedCall*conv2d_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????66 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_312966?
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall+max_pooling2d_16/PartitionedCall_1:output:0conv2d_34_313100conv2d_34_313102*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313099?
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0batch_normalization_8_313105batch_normalization_8_313107batch_normalization_8_313109batch_normalization_8_313111*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_313003?
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0conv2d_35_313126conv2d_35_313128*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313125?
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_312978?
flatten_8/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_8_layer_call_and_return_conditional_losses_313138?
 dense_16/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_16_313152dense_16_313154*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_313151?
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_313168dense_17_313170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_313167x
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp.^batch_normalization_8/StatefulPartitionedCall"^conv2d_32/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2F
!conv2d_32/StatefulPartitionedCall!conv2d_32/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_8_layer_call_fn_314011

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_313034?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
K
#__inference__update_step_xla_313885
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
&__inference_cnn_8_layer_call_fn_313209
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_cnn_8_layer_call_and_return_conditional_losses_313174o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
$__inference_signature_wrapper_313599
input_1!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:
??

unknown_12:	?

unknown_13:	?

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_312957o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
a
E__inference_flatten_8_layer_call_and_return_conditional_losses_313138

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_314069

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?V
?
!__inference__wrapped_model_312957
input_1H
.cnn_8_conv2d_32_conv2d_readvariableop_resource: =
/cnn_8_conv2d_32_biasadd_readvariableop_resource: H
.cnn_8_conv2d_33_conv2d_readvariableop_resource:  =
/cnn_8_conv2d_33_biasadd_readvariableop_resource: H
.cnn_8_conv2d_34_conv2d_readvariableop_resource:  =
/cnn_8_conv2d_34_biasadd_readvariableop_resource: A
3cnn_8_batch_normalization_8_readvariableop_resource: C
5cnn_8_batch_normalization_8_readvariableop_1_resource: R
Dcnn_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource: T
Fcnn_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource: H
.cnn_8_conv2d_35_conv2d_readvariableop_resource:  =
/cnn_8_conv2d_35_biasadd_readvariableop_resource: A
-cnn_8_dense_16_matmul_readvariableop_resource:
??=
.cnn_8_dense_16_biasadd_readvariableop_resource:	?@
-cnn_8_dense_17_matmul_readvariableop_resource:	?<
.cnn_8_dense_17_biasadd_readvariableop_resource:
identity??;cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp?=cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1?*cnn_8/batch_normalization_8/ReadVariableOp?,cnn_8/batch_normalization_8/ReadVariableOp_1?&cnn_8/conv2d_32/BiasAdd/ReadVariableOp?%cnn_8/conv2d_32/Conv2D/ReadVariableOp?&cnn_8/conv2d_33/BiasAdd/ReadVariableOp?%cnn_8/conv2d_33/Conv2D/ReadVariableOp?&cnn_8/conv2d_34/BiasAdd/ReadVariableOp?%cnn_8/conv2d_34/Conv2D/ReadVariableOp?&cnn_8/conv2d_35/BiasAdd/ReadVariableOp?%cnn_8/conv2d_35/Conv2D/ReadVariableOp?%cnn_8/dense_16/BiasAdd/ReadVariableOp?$cnn_8/dense_16/MatMul/ReadVariableOp?%cnn_8/dense_17/BiasAdd/ReadVariableOp?$cnn_8/dense_17/MatMul/ReadVariableOp?
%cnn_8/conv2d_32/Conv2D/ReadVariableOpReadVariableOp.cnn_8_conv2d_32_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
cnn_8/conv2d_32/Conv2DConv2Dinput_1-cnn_8/conv2d_32/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
?
&cnn_8/conv2d_32/BiasAdd/ReadVariableOpReadVariableOp/cnn_8_conv2d_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_8/conv2d_32/BiasAddBiasAddcnn_8/conv2d_32/Conv2D:output:0.cnn_8/conv2d_32/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? z
cnn_8/conv2d_32/ReluRelu cnn_8/conv2d_32/BiasAdd:output:0*
T0*1
_output_shapes
:??????????? ?
cnn_8/max_pooling2d_16/MaxPoolMaxPool"cnn_8/conv2d_32/Relu:activations:0*/
_output_shapes
:?????????oo *
ksize
*
paddingVALID*
strides
?
%cnn_8/conv2d_33/Conv2D/ReadVariableOpReadVariableOp.cnn_8_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
cnn_8/conv2d_33/Conv2DConv2D'cnn_8/max_pooling2d_16/MaxPool:output:0-cnn_8/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm *
paddingVALID*
strides
?
&cnn_8/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp/cnn_8_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_8/conv2d_33/BiasAddBiasAddcnn_8/conv2d_33/Conv2D:output:0.cnn_8/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????mm x
cnn_8/conv2d_33/ReluRelu cnn_8/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:?????????mm ?
 cnn_8/max_pooling2d_16/MaxPool_1MaxPool"cnn_8/conv2d_33/Relu:activations:0*/
_output_shapes
:?????????66 *
ksize
*
paddingVALID*
strides
?
%cnn_8/conv2d_34/Conv2D/ReadVariableOpReadVariableOp.cnn_8_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
cnn_8/conv2d_34/Conv2DConv2D)cnn_8/max_pooling2d_16/MaxPool_1:output:0-cnn_8/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
&cnn_8/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp/cnn_8_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_8/conv2d_34/BiasAddBiasAddcnn_8/conv2d_34/Conv2D:output:0.cnn_8/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? x
cnn_8/conv2d_34/ReluRelu cnn_8/conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
*cnn_8/batch_normalization_8/ReadVariableOpReadVariableOp3cnn_8_batch_normalization_8_readvariableop_resource*
_output_shapes
: *
dtype0?
,cnn_8/batch_normalization_8/ReadVariableOp_1ReadVariableOp5cnn_8_batch_normalization_8_readvariableop_1_resource*
_output_shapes
: *
dtype0?
;cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpDcnn_8_batch_normalization_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
=cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFcnn_8_batch_normalization_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
,cnn_8/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3"cnn_8/conv2d_34/Relu:activations:02cnn_8/batch_normalization_8/ReadVariableOp:value:04cnn_8/batch_normalization_8/ReadVariableOp_1:value:0Ccnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Ecnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
%cnn_8/conv2d_35/Conv2D/ReadVariableOpReadVariableOp.cnn_8_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
cnn_8/conv2d_35/Conv2DConv2D0cnn_8/batch_normalization_8/FusedBatchNormV3:y:0-cnn_8/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
?
&cnn_8/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp/cnn_8_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
cnn_8/conv2d_35/BiasAddBiasAddcnn_8/conv2d_35/Conv2D:output:0.cnn_8/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? x
cnn_8/conv2d_35/ReluRelu cnn_8/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:????????? ?
cnn_8/max_pooling2d_17/MaxPoolMaxPool"cnn_8/conv2d_35/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
f
cnn_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
cnn_8/flatten_8/ReshapeReshape'cnn_8/max_pooling2d_17/MaxPool:output:0cnn_8/flatten_8/Const:output:0*
T0*(
_output_shapes
:???????????
$cnn_8/dense_16/MatMul/ReadVariableOpReadVariableOp-cnn_8_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
cnn_8/dense_16/MatMulMatMul cnn_8/flatten_8/Reshape:output:0,cnn_8/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
%cnn_8/dense_16/BiasAdd/ReadVariableOpReadVariableOp.cnn_8_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
cnn_8/dense_16/BiasAddBiasAddcnn_8/dense_16/MatMul:product:0-cnn_8/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????o
cnn_8/dense_16/ReluRelucnn_8/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
$cnn_8/dense_17/MatMul/ReadVariableOpReadVariableOp-cnn_8_dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
cnn_8/dense_17/MatMulMatMul!cnn_8/dense_16/Relu:activations:0,cnn_8/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
%cnn_8/dense_17/BiasAdd/ReadVariableOpReadVariableOp.cnn_8_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
cnn_8/dense_17/BiasAddBiasAddcnn_8/dense_17/MatMul:product:0-cnn_8/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????n
IdentityIdentitycnn_8/dense_17/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp<^cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp>^cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1+^cnn_8/batch_normalization_8/ReadVariableOp-^cnn_8/batch_normalization_8/ReadVariableOp_1'^cnn_8/conv2d_32/BiasAdd/ReadVariableOp&^cnn_8/conv2d_32/Conv2D/ReadVariableOp'^cnn_8/conv2d_33/BiasAdd/ReadVariableOp&^cnn_8/conv2d_33/Conv2D/ReadVariableOp'^cnn_8/conv2d_34/BiasAdd/ReadVariableOp&^cnn_8/conv2d_34/Conv2D/ReadVariableOp'^cnn_8/conv2d_35/BiasAdd/ReadVariableOp&^cnn_8/conv2d_35/Conv2D/ReadVariableOp&^cnn_8/dense_16/BiasAdd/ReadVariableOp%^cnn_8/dense_16/MatMul/ReadVariableOp&^cnn_8/dense_17/BiasAdd/ReadVariableOp%^cnn_8/dense_17/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:???????????: : : : : : : : : : : : : : : : 2z
;cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp;cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp2~
=cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1=cnn_8/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_12X
*cnn_8/batch_normalization_8/ReadVariableOp*cnn_8/batch_normalization_8/ReadVariableOp2\
,cnn_8/batch_normalization_8/ReadVariableOp_1,cnn_8/batch_normalization_8/ReadVariableOp_12P
&cnn_8/conv2d_32/BiasAdd/ReadVariableOp&cnn_8/conv2d_32/BiasAdd/ReadVariableOp2N
%cnn_8/conv2d_32/Conv2D/ReadVariableOp%cnn_8/conv2d_32/Conv2D/ReadVariableOp2P
&cnn_8/conv2d_33/BiasAdd/ReadVariableOp&cnn_8/conv2d_33/BiasAdd/ReadVariableOp2N
%cnn_8/conv2d_33/Conv2D/ReadVariableOp%cnn_8/conv2d_33/Conv2D/ReadVariableOp2P
&cnn_8/conv2d_34/BiasAdd/ReadVariableOp&cnn_8/conv2d_34/BiasAdd/ReadVariableOp2N
%cnn_8/conv2d_34/Conv2D/ReadVariableOp%cnn_8/conv2d_34/Conv2D/ReadVariableOp2P
&cnn_8/conv2d_35/BiasAdd/ReadVariableOp&cnn_8/conv2d_35/BiasAdd/ReadVariableOp2N
%cnn_8/conv2d_35/Conv2D/ReadVariableOp%cnn_8/conv2d_35/Conv2D/ReadVariableOp2N
%cnn_8/dense_16/BiasAdd/ReadVariableOp%cnn_8/dense_16/BiasAdd/ReadVariableOp2L
$cnn_8/dense_16/MatMul/ReadVariableOp$cnn_8/dense_16/MatMul/ReadVariableOp2N
%cnn_8/dense_17/BiasAdd/ReadVariableOp%cnn_8/dense_17/BiasAdd/ReadVariableOp2L
$cnn_8/dense_17/MatMul/ReadVariableOp$cnn_8/dense_17/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
c
E__inference_dropout_8_layer_call_and_return_conditional_losses_314074

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313099

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????66 : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????66 
 
_user_specified_nameinputs
?
F
*__inference_dropout_8_layer_call_fn_314052

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_8_layer_call_and_return_conditional_losses_313418h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313063

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:??????????? k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_8_layer_call_fn_313998

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_313003?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
K
#__inference__update_step_xla_313845
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?

d
E__inference_dropout_8_layer_call_and_return_conditional_losses_313355

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UU??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ?
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:????????? i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
P
#__inference__update_step_xla_313880
gradient
variable:	?*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	?: *
	_noinline(:I E

_output_shapes
:	?
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
?
?
*__inference_conv2d_33_layer_call_fn_313914

inputs!
unknown:  
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????mm *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313081w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????mm `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????oo : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????oo 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_314047

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
K
#__inference__update_step_xla_313860
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable"?
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2
	
conv3
	conv4
	pool1
	pool4
	batchnorm
	dropout40
flatten
d128
d7

checkpoint
	optimizer

signatures"
_tf_keras_model
?
0
1
2
3
4
5
6
7
8
9
 10
!11
"12
#13
$14
%15"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
"10
#11
$12
%13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
+trace_0
,trace_1
-trace_2
.trace_32?
&__inference_cnn_8_layer_call_fn_313209
&__inference_cnn_8_layer_call_fn_313636
&__inference_cnn_8_layer_call_fn_313673
&__inference_cnn_8_layer_call_fn_313463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z+trace_0z,trace_1z-trace_2z.trace_3
?
/trace_0
0trace_1
1trace_2
2trace_32?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313737
A__inference_cnn_8_layer_call_and_return_conditional_losses_313809
A__inference_cnn_8_layer_call_and_return_conditional_losses_313510
A__inference_cnn_8_layer_call_and_return_conditional_losses_313558?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 z/trace_0z0trace_1z1trace_2z2trace_3
?B?
!__inference__wrapped_model_312957input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses

kernel
bias
 9_jit_compiled_convolution_op"
_tf_keras_layer
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses

kernel
bias
 @_jit_compiled_convolution_op"
_tf_keras_layer
?
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

kernel
bias
 G_jit_compiled_convolution_op"
_tf_keras_layer
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses

kernel
bias
 N_jit_compiled_convolution_op"
_tf_keras_layer
?
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
?
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
?
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
aaxis
	gamma
beta
 moving_mean
!moving_variance"
_tf_keras_layer
?
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses
h_random_generator"
_tf_keras_layer
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses"
_tf_keras_layer
?
o	variables
ptrainable_variables
qregularization_losses
r	keras_api
s__call__
*t&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
?
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
9
model
{save_counter"
_generic_user_object
?
|
_variables
}_iterations
~_learning_rate
_index_dict
?
_momentums
?_velocities
?_update_step_xla"
experimentalOptimizer
-
?serving_default"
signature_map
*:( 2conv2d_32/kernel
: 2conv2d_32/bias
*:(  2conv2d_33/kernel
: 2conv2d_33/bias
*:(  2conv2d_34/kernel
: 2conv2d_34/bias
*:(  2conv2d_35/kernel
: 2conv2d_35/bias
):' 2batch_normalization_8/gamma
(:& 2batch_normalization_8/beta
1:/  (2!batch_normalization_8/moving_mean
5:3  (2%batch_normalization_8/moving_variance
#:!
??2dense_16/kernel
:?2dense_16/bias
": 	?2dense_17/kernel
:2dense_17/bias
.
 0
!1"
trackable_list_wrapper
n
0
	1

2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_cnn_8_layer_call_fn_313209input_1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
&__inference_cnn_8_layer_call_fn_313636inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
&__inference_cnn_8_layer_call_fn_313673inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
&__inference_cnn_8_layer_call_fn_313463input_1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313737inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313809inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313510input_1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313558input_1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_conv2d_32_layer_call_fn_313894?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313905?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_conv2d_33_layer_call_fn_313914?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313925?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_conv2d_34_layer_call_fn_313934?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313945?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_conv2d_35_layer_call_fn_313954?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313965?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling2d_16_layer_call_fn_313970?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_313975?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
1__inference_max_pooling2d_17_layer_call_fn_313980?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_313985?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
0
1
 2
!3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_8_layer_call_fn_313998
6__inference_batch_normalization_8_layer_call_fn_314011?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_314029
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_314047?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_8_layer_call_fn_314052
*__inference_dropout_8_layer_call_fn_314057?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_8_layer_call_and_return_conditional_losses_314069
E__inference_dropout_8_layer_call_and_return_conditional_losses_314074?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_flatten_8_layer_call_fn_314079?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_flatten_8_layer_call_and_return_conditional_losses_314085?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
o	variables
ptrainable_variables
qregularization_losses
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_dense_16_layer_call_fn_314094?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_dense_16_layer_call_and_return_conditional_losses_314105?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_dense_17_layer_call_fn_314114?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_dense_17_layer_call_and_return_conditional_losses_314124?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
:	 2save_counter
?
}0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13"
trackable_list_wrapper
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13"
trackable_list_wrapper
?
?trace_0
?trace_1
?trace_2
?trace_3
?trace_4
?trace_5
?trace_6
?trace_7
?trace_8
?trace_9
?trace_10
?trace_11
?trace_12
?trace_132?
#__inference__update_step_xla_313820
#__inference__update_step_xla_313825
#__inference__update_step_xla_313830
#__inference__update_step_xla_313835
#__inference__update_step_xla_313840
#__inference__update_step_xla_313845
#__inference__update_step_xla_313850
#__inference__update_step_xla_313855
#__inference__update_step_xla_313860
#__inference__update_step_xla_313865
#__inference__update_step_xla_313870
#__inference__update_step_xla_313875
#__inference__update_step_xla_313880
#__inference__update_step_xla_313885?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0z?trace_0z?trace_1z?trace_2z?trace_3z?trace_4z?trace_5z?trace_6z?trace_7z?trace_8z?trace_9z?trace_10z?trace_11z?trace_12z?trace_13
?B?
$__inference_signature_wrapper_313599input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
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
?B?
*__inference_conv2d_32_layer_call_fn_313894inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313905inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
*__inference_conv2d_33_layer_call_fn_313914inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313925inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
*__inference_conv2d_34_layer_call_fn_313934inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313945inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
*__inference_conv2d_35_layer_call_fn_313954inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313965inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
1__inference_max_pooling2d_16_layer_call_fn_313970inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_313975inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
1__inference_max_pooling2d_17_layer_call_fn_313980inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_313985inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_8_layer_call_fn_313998inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
6__inference_batch_normalization_8_layer_call_fn_314011inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_314029inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_314047inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
*__inference_dropout_8_layer_call_fn_314052inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
*__inference_dropout_8_layer_call_fn_314057inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dropout_8_layer_call_and_return_conditional_losses_314069inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dropout_8_layer_call_and_return_conditional_losses_314074inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
*__inference_flatten_8_layer_call_fn_314079inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_flatten_8_layer_call_and_return_conditional_losses_314085inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dense_16_layer_call_fn_314094inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dense_16_layer_call_and_return_conditional_losses_314105inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
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
?B?
)__inference_dense_17_layer_call_fn_314114inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_dense_17_layer_call_and_return_conditional_losses_314124inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
/:- 2Adam/m/conv2d_32/kernel
/:- 2Adam/v/conv2d_32/kernel
!: 2Adam/m/conv2d_32/bias
!: 2Adam/v/conv2d_32/bias
/:-  2Adam/m/conv2d_33/kernel
/:-  2Adam/v/conv2d_33/kernel
!: 2Adam/m/conv2d_33/bias
!: 2Adam/v/conv2d_33/bias
/:-  2Adam/m/conv2d_34/kernel
/:-  2Adam/v/conv2d_34/kernel
!: 2Adam/m/conv2d_34/bias
!: 2Adam/v/conv2d_34/bias
/:-  2Adam/m/conv2d_35/kernel
/:-  2Adam/v/conv2d_35/kernel
!: 2Adam/m/conv2d_35/bias
!: 2Adam/v/conv2d_35/bias
.:, 2"Adam/m/batch_normalization_8/gamma
.:, 2"Adam/v/batch_normalization_8/gamma
-:+ 2!Adam/m/batch_normalization_8/beta
-:+ 2!Adam/v/batch_normalization_8/beta
(:&
??2Adam/m/dense_16/kernel
(:&
??2Adam/v/dense_16/kernel
!:?2Adam/m/dense_16/bias
!:?2Adam/v/dense_16/bias
':%	?2Adam/m/dense_17/kernel
':%	?2Adam/v/dense_17/kernel
 :2Adam/m/dense_17/bias
 :2Adam/v/dense_17/bias
?B?
#__inference__update_step_xla_313820gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313825gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313830gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313835gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313840gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313845gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313850gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313855gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313860gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313865gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313870gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313875gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313880gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference__update_step_xla_313885gradientvariable"?
???
FullArgSpec2
args*?'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count?
#__inference__update_step_xla_313820~x?u
n?k
!?
gradient 
<?9	%?"
? 
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313825f`?]
V?S
?
gradient 
0?-	?
? 
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313830~x?u
n?k
!?
gradient  
<?9	%?"
?  
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313835f`?]
V?S
?
gradient 
0?-	?
? 
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313840~x?u
n?k
!?
gradient  
<?9	%?"
?  
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313845f`?]
V?S
?
gradient 
0?-	?
? 
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313850~x?u
n?k
!?
gradient  
<?9	%?"
?  
?
p
` VariableSpec 
`?קƵ??
? "
 ?
#__inference__update_step_xla_313855f`?]
V?S
?
gradient 
0?-	?
? 
?
p
` VariableSpec 
`?ԧƵ??
? "
 ?
#__inference__update_step_xla_313860f`?]
V?S
?
gradient 
0?-	?
? 
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313865f`?]
V?S
?
gradient 
0?-	?
? 
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313870rl?i
b?_
?
gradient
??
6?3	?
?
??
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313875hb?_
X?U
?
gradient?
1?.	?
??
?
p
` VariableSpec 
`???Ƶ??
? "
 ?
#__inference__update_step_xla_313880pj?g
`?]
?
gradient	?
5?2	?
?	?
?
p
` VariableSpec 
`?ҩƵ??
? "
 ?
#__inference__update_step_xla_313885f`?]
V?S
?
gradient
0?-	?
?
?
p
` VariableSpec 
`?ɩƵ??
? "
 ?
!__inference__wrapped_model_312957? !"#$%:?7
0?-
+?(
input_1???????????
? "3?0
.
output_1"?
output_1??????????
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_314029? !M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "F?C
<?9
tensor_0+??????????????????????????? 
? ?
Q__inference_batch_normalization_8_layer_call_and_return_conditional_losses_314047? !M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "F?C
<?9
tensor_0+??????????????????????????? 
? ?
6__inference_batch_normalization_8_layer_call_fn_313998? !M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? ";?8
unknown+??????????????????????????? ?
6__inference_batch_normalization_8_layer_call_fn_314011? !M?J
C?@
:?7
inputs+??????????????????????????? 
p
? ";?8
unknown+??????????????????????????? ?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313510? !"#$%J?G
0?-
+?(
input_1???????????
?

trainingp ",?)
"?
tensor_0?????????
? ?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313558? !"#$%J?G
0?-
+?(
input_1???????????
?

trainingp",?)
"?
tensor_0?????????
? ?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313737? !"#$%I?F
/?,
*?'
inputs???????????
?

trainingp ",?)
"?
tensor_0?????????
? ?
A__inference_cnn_8_layer_call_and_return_conditional_losses_313809? !"#$%I?F
/?,
*?'
inputs???????????
?

trainingp",?)
"?
tensor_0?????????
? ?
&__inference_cnn_8_layer_call_fn_313209? !"#$%J?G
0?-
+?(
input_1???????????
?

trainingp "!?
unknown??????????
&__inference_cnn_8_layer_call_fn_313463? !"#$%J?G
0?-
+?(
input_1???????????
?

trainingp"!?
unknown??????????
&__inference_cnn_8_layer_call_fn_313636? !"#$%I?F
/?,
*?'
inputs???????????
?

trainingp "!?
unknown??????????
&__inference_cnn_8_layer_call_fn_313673? !"#$%I?F
/?,
*?'
inputs???????????
?

trainingp"!?
unknown??????????
E__inference_conv2d_32_layer_call_and_return_conditional_losses_313905w9?6
/?,
*?'
inputs???????????
? "6?3
,?)
tensor_0??????????? 
? ?
*__inference_conv2d_32_layer_call_fn_313894l9?6
/?,
*?'
inputs???????????
? "+?(
unknown??????????? ?
E__inference_conv2d_33_layer_call_and_return_conditional_losses_313925s7?4
-?*
(?%
inputs?????????oo 
? "4?1
*?'
tensor_0?????????mm 
? ?
*__inference_conv2d_33_layer_call_fn_313914h7?4
-?*
(?%
inputs?????????oo 
? ")?&
unknown?????????mm ?
E__inference_conv2d_34_layer_call_and_return_conditional_losses_313945s7?4
-?*
(?%
inputs?????????66 
? "4?1
*?'
tensor_0????????? 
? ?
*__inference_conv2d_34_layer_call_fn_313934h7?4
-?*
(?%
inputs?????????66 
? ")?&
unknown????????? ?
E__inference_conv2d_35_layer_call_and_return_conditional_losses_313965s7?4
-?*
(?%
inputs????????? 
? "4?1
*?'
tensor_0????????? 
? ?
*__inference_conv2d_35_layer_call_fn_313954h7?4
-?*
(?%
inputs????????? 
? ")?&
unknown????????? ?
D__inference_dense_16_layer_call_and_return_conditional_losses_314105e"#0?-
&?#
!?
inputs??????????
? "-?*
#? 
tensor_0??????????
? ?
)__inference_dense_16_layer_call_fn_314094Z"#0?-
&?#
!?
inputs??????????
? ""?
unknown???????????
D__inference_dense_17_layer_call_and_return_conditional_losses_314124d$%0?-
&?#
!?
inputs??????????
? ",?)
"?
tensor_0?????????
? ?
)__inference_dense_17_layer_call_fn_314114Y$%0?-
&?#
!?
inputs??????????
? "!?
unknown??????????
E__inference_dropout_8_layer_call_and_return_conditional_losses_314069s;?8
1?.
(?%
inputs????????? 
p
? "4?1
*?'
tensor_0????????? 
? ?
E__inference_dropout_8_layer_call_and_return_conditional_losses_314074s;?8
1?.
(?%
inputs????????? 
p 
? "4?1
*?'
tensor_0????????? 
? ?
*__inference_dropout_8_layer_call_fn_314052h;?8
1?.
(?%
inputs????????? 
p 
? ")?&
unknown????????? ?
*__inference_dropout_8_layer_call_fn_314057h;?8
1?.
(?%
inputs????????? 
p
? ")?&
unknown????????? ?
E__inference_flatten_8_layer_call_and_return_conditional_losses_314085h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
tensor_0??????????
? ?
*__inference_flatten_8_layer_call_fn_314079]7?4
-?*
(?%
inputs????????? 
? ""?
unknown???????????
L__inference_max_pooling2d_16_layer_call_and_return_conditional_losses_313975?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
1__inference_max_pooling2d_16_layer_call_fn_313970?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
L__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_313985?R?O
H?E
C?@
inputs4????????????????????????????????????
? "O?L
E?B
tensor_04????????????????????????????????????
? ?
1__inference_max_pooling2d_17_layer_call_fn_313980?R?O
H?E
C?@
inputs4????????????????????????????????????
? "D?A
unknown4?????????????????????????????????????
$__inference_signature_wrapper_313599? !"#$%E?B
? 
;?8
6
input_1+?(
input_1???????????"3?0
.
output_1"?
output_1?????????