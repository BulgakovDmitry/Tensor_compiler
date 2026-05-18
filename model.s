	.text
	.file	"tensor_network"
	.globl	tensorCompForwardImpl
	.p2align	4, 0x90
	.type	tensorCompForwardImpl,@function
tensorCompForwardImpl:
	.cfi_startproc
	xorl	%eax, %eax
	retq
.Lfunc_end0:
	.size	tensorCompForwardImpl, .Lfunc_end0-tensorCompForwardImpl
	.cfi_endproc

	.section	".note.GNU-stack","",@progbits
