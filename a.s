	.text
	.file	"tensor_network"
	.globl	SingleRelu
	.p2align	4, 0x90
	.type	SingleRelu,@function
SingleRelu:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rsi, %rbx
	movq	56(%rsp), %r14
	movq	48(%rsp), %r15
	movl	$72, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	leaq	.L__constant_1x2xf32(%rip), %rcx
	xorl	%edx, %edx
	movq	%rax, %rsi
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_9:
	incq	%rdx
	addq	$8, %rsi
	addq	$8, %rcx
	addq	$8, %rbx
.LBB0_1:
	testq	%rdx, %rdx
	jg	.LBB0_10
	xorl	%edi, %edi
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_8:
	movdqa	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rsi,%rdi,4)
	incq	%rdi
.LBB0_3:
	cmpq	$1, %rdi
	jg	.LBB0_9
	movd	(%rbx,%rdi,4), %xmm2
	movd	(%rcx,%rdi,4), %xmm0
	movd	%xmm2, %r8d
	testl	%r8d, %r8d
	movdqa	%xmm2, %xmm1
	js	.LBB0_6
	movdqa	%xmm0, %xmm1
.LBB0_6:
	js	.LBB0_8
	movdqa	%xmm2, %xmm0
	jmp	.LBB0_8
.LBB0_10:
	movq	(%rax), %rax
	movq	%rax, (%r15,%r14,4)
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	SingleRelu, .Lfunc_end0-SingleRelu
	.cfi_endproc

	.type	.L__constant_1x2xf32,@object
	.section	.rodata,"a",@progbits
	.p2align	2, 0x0
.L__constant_1x2xf32:
	.zero	8
	.size	.L__constant_1x2xf32, 8

	.section	".note.GNU-stack","",@progbits
