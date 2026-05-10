	.att_syntax
	.file	"tensor_network"
	.text
	.globl	tensorCompForwardImpl
	.p2align	4
	.type	tensorCompForwardImpl,@function
tensorCompForwardImpl:
	.cfi_startproc
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	movq	%rdi, %rax
	movq	32(%rsp), %rdi
	movq	24(%rsp), %r10
	xorl	%r11d, %r11d
	xorps	%xmm0, %xmm0
	movq	%rdx, %rbx
	jmp	.LBB0_1
	.p2align	4
.LBB0_5:
	incq	%r11
	addq	$8, %rbx
.LBB0_1:
	testq	%r11, %r11
	jg	.LBB0_6
	xorl	%r14d, %r14d
	cmpq	$1, %r14
	jg	.LBB0_5
	.p2align	4
.LBB0_4:
	movss	(%rbx,%r14,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rbx,%r14,4)
	incq	%r14
	cmpq	$1, %r14
	jle	.LBB0_4
	jmp	.LBB0_5
.LBB0_6:
	movq	%rsi, (%rax)
	movq	%rdx, 8(%rax)
	movq	%rcx, 16(%rax)
	movq	%r8, 24(%rax)
	movq	%r9, 32(%rax)
	movq	%r10, 40(%rax)
	movq	%rdi, 48(%rax)
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	tensorCompForwardImpl, .Lfunc_end0-tensorCompForwardImpl
	.cfi_endproc

	.type	.L__constant_1x2xf32,@object
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.L__constant_1x2xf32:
	.zero	8
	.size	.L__constant_1x2xf32, 8

	.section	".note.GNU-stack","",@progbits
