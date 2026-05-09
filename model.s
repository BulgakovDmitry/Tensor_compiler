	.att_syntax
	.file	"tensor_network"
	.text
	.globl	tensorCompForwardImpl
	.p2align	4
	.type	tensorCompForwardImpl,@function
tensorCompForwardImpl:
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
	movq	%rdi, %rax
	movq	56(%rsp), %r11
	movq	40(%rsp), %rdi
	movq	32(%rsp), %r10
	xorl	%ebx, %ebx
	movq	%rdx, %r14
	jmp	.LBB0_1
	.p2align	4
.LBB0_5:
	incq	%rbx
	addq	$8, %r11
	addq	$8, %r14
.LBB0_1:
	cmpq	$1, %rbx
	jg	.LBB0_6
	xorl	%r15d, %r15d
	cmpq	$1, %r15
	jg	.LBB0_5
	.p2align	4
.LBB0_4:
	movss	(%r14,%r15,4), %xmm0
	mulss	(%r11,%r15,4), %xmm0
	movss	%xmm0, (%r14,%r15,4)
	incq	%r15
	cmpq	$1, %r15
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
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end0:
	.size	tensorCompForwardImpl, .Lfunc_end0-tensorCompForwardImpl
	.cfi_endproc

	.section	".note.GNU-stack","",@progbits
