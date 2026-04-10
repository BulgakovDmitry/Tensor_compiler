	.text
	.file	"tensor_network"
	.globl	TwoInputMulThenAddConstGraph
	.p2align	4, 0x90
	.type	TwoInputMulThenAddConstGraph,@function
TwoInputMulThenAddConstGraph:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rsi, %r14
	movq	128(%rsp), %r15
	movq	120(%rsp), %r12
	movq	64(%rsp), %r13
	movl	$80, %edi
	callq	malloc@PLT
	movq	%rax, %rbx
	addq	$63, %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_5:
	incq	%rax
	addq	$8, %rcx
	addq	$8, %r13
	addq	$8, %r14
.LBB0_1:
	cmpq	$1, %rax
	jg	.LBB0_6
	xorl	%edx, %edx
	cmpq	$1, %rdx
	jg	.LBB0_5
	.p2align	4, 0x90
.LBB0_4:
	movss	(%r14,%rdx,4), %xmm0
	mulss	(%r13,%rdx,4), %xmm0
	movss	%xmm0, (%rcx,%rdx,4)
	incq	%rdx
	cmpq	$1, %rdx
	jle	.LBB0_4
	jmp	.LBB0_5
.LBB0_6:
	movl	$80, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	leaq	.L__constant_2x2xf32(%rip), %rcx
	xorl	%edx, %edx
	movq	%rax, %rsi
	jmp	.LBB0_7
	.p2align	4, 0x90
.LBB0_11:
	incq	%rdx
	addq	$8, %rsi
	addq	$8, %rcx
	addq	$8, %rbx
.LBB0_7:
	cmpq	$1, %rdx
	jg	.LBB0_12
	xorl	%edi, %edi
	cmpq	$1, %rdi
	jg	.LBB0_11
	.p2align	4, 0x90
.LBB0_10:
	movss	(%rbx,%rdi,4), %xmm0
	addss	(%rcx,%rdi,4), %xmm0
	movss	%xmm0, (%rsi,%rdi,4)
	incq	%rdi
	cmpq	$1, %rdi
	jle	.LBB0_10
	jmp	.LBB0_11
.LBB0_12:
	movq	(%rax), %rcx
	movq	8(%rax), %rax
	movq	%rax, 8(%r12,%r15,4)
	movq	%rcx, (%r12,%r15,4)
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	TwoInputMulThenAddConstGraph, .Lfunc_end0-TwoInputMulThenAddConstGraph
	.cfi_endproc

	.type	.L__constant_2x2xf32,@object
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
.L__constant_2x2xf32:
	.long	0x3f800000
	.long	0xc0000000
	.long	0x3f000000
	.long	0x40400000
	.size	.L__constant_2x2xf32, 16

	.section	".note.GNU-stack","",@progbits
