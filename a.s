	.text
	.file	"tensor_network"
	.globl	TwoInputMulGraph
	.p2align	4, 0x90
	.type	TwoInputMulGraph,@function
TwoInputMulGraph:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rsi, %rbx
	movq	128(%rsp), %r14
	movq	120(%rsp), %r15
	movq	64(%rsp), %r12
	movl	$80, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ecx, %ecx
	movq	%rax, %rdx
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_5:
	incq	%rcx
	addq	$8, %rdx
	addq	$8, %r12
	addq	$8, %rbx
.LBB0_1:
	cmpq	$1, %rcx
	jg	.LBB0_6
	xorl	%esi, %esi
	cmpq	$1, %rsi
	jg	.LBB0_5
	.p2align	4, 0x90
.LBB0_4:
	movss	(%rbx,%rsi,4), %xmm0
	mulss	(%r12,%rsi,4), %xmm0
	movss	%xmm0, (%rdx,%rsi,4)
	incq	%rsi
	cmpq	$1, %rsi
	jle	.LBB0_4
	jmp	.LBB0_5
.LBB0_6:
	movq	(%rax), %rcx
	movq	8(%rax), %rax
	movq	%rax, 8(%r15,%r14,4)
	movq	%rcx, (%r15,%r14,4)
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	TwoInputMulGraph, .Lfunc_end0-TwoInputMulGraph
	.cfi_endproc

	.section	".note.GNU-stack","",@progbits
