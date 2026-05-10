	.text
	.file	"tensor_network"
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0
.LCPI0_0:
	.long	0x3ca72f05
.LCPI0_1:
	.long	0x3727f09f
	.text
	.globl	tensorCompForwardImpl
	.p2align	4, 0x90
	.type	tensorCompForwardImpl,@function
tensorCompForwardImpl:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$152, %rsp
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movq	%r9, %rbx
	movq	%rcx, %r13
	movq	%rdx, %r15
	movq	%rsi, %r14
	movq	%rdi, -152(%rbp)
	movq	16(%rbp), %rcx
	movq	24(%rbp), %rax
	movq	%rax, -160(%rbp)
	movq	%rcx, -168(%rbp)
	movq	%r9, -176(%rbp)
	movq	%r8, -184(%rbp)
	movq	%r8, -48(%rbp)
	imulq	$634800, %r8, %rdi
	addq	$64, %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -64(%rbp)
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_11:
	incq	%rcx
	addq	$634800, %rdx
.LBB0_1:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_12
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_10:
	incq	%rdi
	addq	$211600, %rsi
.LBB0_3:
	cmpq	$2, %rdi
	jg	.LBB0_11
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_5
	.p2align	4, 0x90
.LBB0_9:
	incq	%r9
	addq	$920, %r8
.LBB0_5:
	cmpq	$229, %r9
	jg	.LBB0_10
	xorl	%r10d, %r10d
	cmpq	$229, %r10
	jg	.LBB0_9
	.p2align	4, 0x90
.LBB0_8:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$229, %r10
	jle	.LBB0_8
	jmp	.LBB0_9
.LBB0_12:
	movq	%rsp, %r12
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movq	16(%rbp), %rsi
	movq	%rsi, -56(%rdx)
	movq	%rbx, -64(%rdx)
	movq	-48(%rbp), %rbx
	movq	%rbx, -72(%rdx)
	movq	%r13, -80(%rdx)
	movq	%r15, -88(%rdx)
	movq	%r14, -96(%rdx)
	movq	24(%rbp), %rsi
	movq	%rsi, -48(%rdx)
	movq	32(%rbp), %rsi
	movq	%rsi, -40(%rdx)
	movq	40(%rbp), %rsi
	movq	%rsi, -32(%rdx)
	movq	48(%rbp), %rsi
	movq	%rsi, -24(%rdx)
	movq	56(%rbp), %rsi
	movq	%rsi, -16(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rbx, -72(%rdx)
	movq	-64(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$230, -24(%rdx)
	movq	$52900, -32(%rdx)
	movq	$158700, -40(%rdx)
	movq	$224, -48(%rdx)
	movq	$224, -56(%rdx)
	movq	$3, -64(%rdx)
	movq	$693, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	%r12, %rsp
	imulq	$3211264, %rbx, %rdi
	orq	$64, %rdi
	movq	%rdi, -104(%rbp)
	callq	malloc@PLT
	movq	%rax, -72(%rbp)
	leaq	63(%rax), %r12
	andq	$-64, %r12
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_13
	.p2align	4, 0x90
.LBB0_23:
	incq	%rax
	addq	$3211264, %rcx
.LBB0_13:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_24
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_15
	.p2align	4, 0x90
.LBB0_22:
	incq	%rsi
	addq	$50176, %rdx
.LBB0_15:
	cmpq	$63, %rsi
	jg	.LBB0_23
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_17
	.p2align	4, 0x90
.LBB0_21:
	incq	%r8
	addq	$448, %rdi
.LBB0_17:
	cmpq	$111, %r8
	jg	.LBB0_22
	xorl	%r9d, %r9d
	cmpq	$111, %r9
	jg	.LBB0_21
	.p2align	4, 0x90
.LBB0_20:
	movl	$0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$111, %r9
	jle	.LBB0_20
	jmp	.LBB0_21
.LBB0_24:
	xorl	%eax, %eax
	jmp	.LBB0_25
	.p2align	4, 0x90
.LBB0_44:
	incq	%rax
	addq	$634800, -64(%rbp)
.LBB0_25:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_45
	movq	1576(%rbp), %rcx
	movq	%rcx, -80(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_27
	.p2align	4, 0x90
.LBB0_43:
	incq	%rdx
	addq	$588, -80(%rbp)
.LBB0_27:
	cmpq	$63, %rdx
	jg	.LBB0_44
	movq	-64(%rbp), %r8
	xorl	%edi, %edi
	jmp	.LBB0_29
	.p2align	4, 0x90
.LBB0_42:
	incq	%rdi
	movq	-56(%rbp), %r8
	addq	$1840, %r8
.LBB0_29:
	cmpq	$111, %rdi
	jg	.LBB0_43
	movq	%r8, -56(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_31
	.p2align	4, 0x90
.LBB0_41:
	incq	%r9
	movq	-88(%rbp), %r8
	addq	$8, %r8
.LBB0_31:
	cmpq	$111, %r9
	jg	.LBB0_42
	movq	%r8, -88(%rbp)
	movq	-80(%rbp), %rcx
	xorl	%esi, %esi
	jmp	.LBB0_33
	.p2align	4, 0x90
.LBB0_40:
	incq	%rsi
	addq	$196, %rcx
	addq	$211600, %r8
.LBB0_33:
	cmpq	$2, %rsi
	jg	.LBB0_41
	movq	%r8, %r10
	movq	%rcx, %r11
	xorl	%r14d, %r14d
	jmp	.LBB0_35
	.p2align	4, 0x90
.LBB0_39:
	incq	%r14
	addq	$28, %r11
	addq	$920, %r10
.LBB0_35:
	cmpq	$6, %r14
	jg	.LBB0_40
	xorl	%r15d, %r15d
	cmpq	$6, %r15
	jg	.LBB0_39
	.p2align	4, 0x90
.LBB0_38:
	movss	(%r10,%r15,4), %xmm0
	imulq	$802816, %rax, %r13
	imulq	$12544, %rdx, %rbx
	addq	%r13, %rbx
	imulq	$112, %rdi, %r13
	addq	%r9, %r13
	addq	%rbx, %r13
	mulss	(%r11,%r15,4), %xmm0
	addss	(%r12,%r13,4), %xmm0
	movss	%xmm0, (%r12,%r13,4)
	incq	%r15
	cmpq	$6, %r15
	jle	.LBB0_38
	jmp	.LBB0_39
.LBB0_45:
	xorl	%eax, %eax
	movss	.LCPI0_1(%rip), %xmm0
	movq	%r12, %rcx
	jmp	.LBB0_46
	.p2align	4, 0x90
.LBB0_56:
	incq	%rax
	addq	$3211264, %rcx
.LBB0_46:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_57
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_48
	.p2align	4, 0x90
.LBB0_55:
	incq	%rsi
	addq	$50176, %rdx
.LBB0_48:
	cmpq	$63, %rsi
	jg	.LBB0_56
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_50
	.p2align	4, 0x90
.LBB0_54:
	incq	%r8
	addq	$448, %rdi
.LBB0_50:
	cmpq	$111, %r8
	jg	.LBB0_55
	xorl	%r9d, %r9d
	cmpq	$111, %r9
	jg	.LBB0_54
	.p2align	4, 0x90
.LBB0_53:
	movss	(%rdi,%r9,4), %xmm1
	movq	4688(%rbp), %r10
	subss	(%r10,%rsi,4), %xmm1
	movq	4608(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm2
	addss	%xmm0, %xmm2
	sqrtss	%xmm2, %xmm2
	divss	%xmm2, %xmm1
	movq	4816(%rbp), %r10
	mulss	(%r10,%rsi,4), %xmm1
	movq	4896(%rbp), %r10
	addss	(%r10,%rsi,4), %xmm1
	movss	%xmm1, (%rdi,%r9,4)
	incq	%r9
	cmpq	$111, %r9
	jle	.LBB0_53
	jmp	.LBB0_54
.LBB0_57:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r12, %rcx
	jmp	.LBB0_58
	.p2align	4, 0x90
.LBB0_68:
	incq	%rax
	addq	$3211264, %rcx
.LBB0_58:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_69
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_60
	.p2align	4, 0x90
.LBB0_67:
	incq	%rsi
	addq	$50176, %rdx
.LBB0_60:
	cmpq	$63, %rsi
	jg	.LBB0_68
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_62
	.p2align	4, 0x90
.LBB0_66:
	incq	%r8
	addq	$448, %rdi
.LBB0_62:
	cmpq	$111, %r8
	jg	.LBB0_67
	xorl	%r9d, %r9d
	cmpq	$111, %r9
	jg	.LBB0_66
	.p2align	4, 0x90
.LBB0_65:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$111, %r9
	jle	.LBB0_65
	jmp	.LBB0_66
.LBB0_69:
	imulq	$3268864, -48(%rbp), %rdi
	orq	$64, %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -56(%rbp)
	jmp	.LBB0_70
	.p2align	4, 0x90
.LBB0_80:
	incq	%rcx
	addq	$3268864, %rdx
.LBB0_70:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_81
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_72
	.p2align	4, 0x90
.LBB0_79:
	incq	%rdi
	addq	$51076, %rsi
.LBB0_72:
	cmpq	$63, %rdi
	jg	.LBB0_80
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_74
	.p2align	4, 0x90
.LBB0_78:
	incq	%r9
	addq	$452, %r8
.LBB0_74:
	cmpq	$112, %r9
	jg	.LBB0_79
	xorl	%r10d, %r10d
	cmpq	$112, %r10
	jg	.LBB0_78
	.p2align	4, 0x90
.LBB0_77:
	movl	$-8388608, (%r8,%r10,4)
	incq	%r10
	cmpq	$112, %r10
	jle	.LBB0_77
	jmp	.LBB0_78
.LBB0_81:
	movq	%rsp, %rbx
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$112, %edi
	movq	%rdi, -56(%rdx)
	movl	$64, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %r14
	movq	%r14, -72(%rdx)
	movq	%r12, -88(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -96(%rdx)
	movq	%rdi, -48(%rdx)
	movl	$802816, %esi
	movq	%rsi, -40(%rdx)
	movl	$12544, %esi
	movq	%rsi, -32(%rdx)
	movq	%rdi, -24(%rdx)
	movl	$1, %esi
	movq	%rsi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%r14, -72(%rdx)
	movq	-56(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$113, -24(%rdx)
	movq	$12769, -32(%rdx)
	movq	$817216, -40(%rdx)
	movq	$112, -48(%rdx)
	movq	$112, -56(%rdx)
	movq	$64, -64(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	%rbx, %rsp
	imulq	$802816, %r14, %r14
	orq	$64, %r14
	movq	%r14, %rdi
	callq	malloc@PLT
	movq	%rax, -96(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%r15d, %r15d
	movq	%r14, -128(%rbp)
	movq	%r14, %rdi
	callq	malloc@PLT
	movq	%rax, %r13
	addq	$63, %r13
	andq	$-64, %r13
	movq	%r13, %rax
	movq	6432(%rbp), %r10
	jmp	.LBB0_82
	.p2align	4, 0x90
.LBB0_92:
	incq	%r15
	addq	$802816, %rax
.LBB0_82:
	cmpq	-48(%rbp), %r15
	jge	.LBB0_93
	movq	%rax, %rcx
	xorl	%edx, %edx
	jmp	.LBB0_84
	.p2align	4, 0x90
.LBB0_91:
	incq	%rdx
	addq	$12544, %rcx
.LBB0_84:
	cmpq	$63, %rdx
	jg	.LBB0_92
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_86
	.p2align	4, 0x90
.LBB0_90:
	incq	%rdi
	addq	$224, %rsi
.LBB0_86:
	cmpq	$55, %rdi
	jg	.LBB0_91
	xorl	%r8d, %r8d
	cmpq	$55, %r8
	jg	.LBB0_90
	.p2align	4, 0x90
.LBB0_89:
	movl	$-8388608, (%rsi,%r8,4)
	incq	%r8
	cmpq	$55, %r8
	jle	.LBB0_89
	jmp	.LBB0_90
.LBB0_93:
	xorl	%ecx, %ecx
	movq	-56(%rbp), %rdx
	jmp	.LBB0_94
	.p2align	4, 0x90
.LBB0_114:
	movq	-64(%rbp), %rcx
	incq	%rcx
	movq	-56(%rbp), %rdx
	addq	$3268864, %rdx
.LBB0_94:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_115
	movq	%rcx, -64(%rbp)
	imulq	$200704, %rcx, %rcx
	movq	%rcx, -80(%rbp)
	movq	%rdx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_96
	.p2align	4, 0x90
.LBB0_113:
	incq	%rsi
	movq	-88(%rbp), %rdx
	addq	$51076, %rdx
.LBB0_96:
	cmpq	$63, %rsi
	jg	.LBB0_114
	imulq	$3136, %rsi, %rdi
	addq	-80(%rbp), %rdi
	movq	%rdx, -88(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_98
	.p2align	4, 0x90
.LBB0_112:
	incq	%r9
	addq	$904, %rdx
	movq	6432(%rbp), %r10
.LBB0_98:
	cmpq	$55, %r9
	jg	.LBB0_113
	imulq	$56, %r9, %r10
	movq	%rdx, %r8
	xorl	%r15d, %r15d
	jmp	.LBB0_100
	.p2align	4, 0x90
.LBB0_111:
	incq	%r15
	addq	$8, %r8
.LBB0_100:
	cmpq	$55, %r15
	jg	.LBB0_112
	leaq	(%r10,%r15), %r12
	addq	%rdi, %r12
	movq	%r8, %r11
	xorl	%eax, %eax
	jmp	.LBB0_102
	.p2align	4, 0x90
.LBB0_110:
	incq	%rax
	addq	$452, %r11
.LBB0_102:
	cmpq	$2, %rax
	jg	.LBB0_111
	xorl	%ecx, %ecx
	jmp	.LBB0_104
	.p2align	4, 0x90
.LBB0_109:
	movdqa	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r13,%r12,4)
	incq	%rcx
.LBB0_104:
	cmpq	$2, %rcx
	jg	.LBB0_110
	movd	(%r11,%rcx,4), %xmm0
	movd	(%r13,%r12,4), %xmm2
	movd	%xmm2, %r14d
	testl	%r14d, %r14d
	movdqa	%xmm2, %xmm1
	js	.LBB0_107
	movdqa	%xmm0, %xmm1
.LBB0_107:
	js	.LBB0_109
	movdqa	%xmm2, %xmm0
	jmp	.LBB0_109
.LBB0_115:
	xorl	%eax, %eax
	movq	%rbx, %rcx
	jmp	.LBB0_116
	.p2align	4, 0x90
.LBB0_126:
	incq	%rax
	addq	$802816, %rcx
.LBB0_116:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_127
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_118
	.p2align	4, 0x90
.LBB0_125:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_118:
	cmpq	$63, %rsi
	jg	.LBB0_126
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_120
	.p2align	4, 0x90
.LBB0_124:
	incq	%r8
	addq	$224, %rdi
.LBB0_120:
	cmpq	$55, %r8
	jg	.LBB0_125
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_124
	.p2align	4, 0x90
.LBB0_123:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_123
	jmp	.LBB0_124
.LBB0_127:
	xorl	%eax, %eax
	movq	%r13, -72(%rbp)
	jmp	.LBB0_128
	.p2align	4, 0x90
.LBB0_147:
	incq	%rax
	addq	$802816, -72(%rbp)
.LBB0_128:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_148
	movq	6512(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_130
	.p2align	4, 0x90
.LBB0_146:
	incq	%rsi
	addq	$256, -56(%rbp)
.LBB0_130:
	cmpq	$63, %rsi
	jg	.LBB0_147
	movq	-72(%rbp), %rcx
	xorl	%r8d, %r8d
	jmp	.LBB0_132
	.p2align	4, 0x90
.LBB0_145:
	incq	%r8
	movq	-64(%rbp), %rcx
	addq	$224, %rcx
.LBB0_132:
	cmpq	$55, %r8
	jg	.LBB0_146
	movq	%rcx, -64(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_134
	.p2align	4, 0x90
.LBB0_144:
	incq	%r10
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_134:
	cmpq	$55, %r10
	jg	.LBB0_145
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r12
	movq	-56(%rbp), %rdx
	xorl	%r15d, %r15d
	jmp	.LBB0_136
	.p2align	4, 0x90
.LBB0_143:
	incq	%r15
	addq	$4, %rdx
	movq	-88(%rbp), %r12
	addq	$12544, %r12
.LBB0_136:
	cmpq	$63, %r15
	jg	.LBB0_144
	movq	%r12, -88(%rbp)
	movq	%rdx, %r14
	xorl	%edi, %edi
	jmp	.LBB0_138
	.p2align	4, 0x90
.LBB0_142:
	incq	%rdi
	addq	$4, %r14
	addq	$224, %r12
.LBB0_138:
	testq	%rdi, %rdi
	jg	.LBB0_143
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_142
	.p2align	4, 0x90
.LBB0_141:
	imulq	$200704, %rax, %r9
	movss	(%r12,%rcx,4), %xmm0
	imulq	$3136, %rsi, %r11
	addq	%r9, %r11
	imulq	$56, %r8, %r9
	addq	%r10, %r9
	addq	%r11, %r9
	mulss	(%r14,%rcx,4), %xmm0
	addss	(%rbx,%r9,4), %xmm0
	movss	%xmm0, (%rbx,%r9,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_141
	jmp	.LBB0_142
.LBB0_148:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_149
	.p2align	4, 0x90
.LBB0_159:
	incq	%rax
	addq	$802816, %rcx
.LBB0_149:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_160
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_151
	.p2align	4, 0x90
.LBB0_158:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_151:
	cmpq	$63, %rsi
	jg	.LBB0_159
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_153
	.p2align	4, 0x90
.LBB0_157:
	incq	%r8
	addq	$224, %rdi
.LBB0_153:
	cmpq	$55, %r8
	jg	.LBB0_158
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_157
	.p2align	4, 0x90
.LBB0_156:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_156
	jmp	.LBB0_157
.LBB0_160:
	imulq	$861184, -48(%rbp), %rdi
	orq	$64, %rdi
	movq	%rdi, -136(%rbp)
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -72(%rbp)
	jmp	.LBB0_161
	.p2align	4, 0x90
.LBB0_171:
	incq	%rcx
	addq	$861184, %rdx
.LBB0_161:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_172
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_163
	.p2align	4, 0x90
.LBB0_170:
	incq	%rdi
	addq	$13456, %rsi
.LBB0_163:
	cmpq	$63, %rdi
	jg	.LBB0_171
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_165
	.p2align	4, 0x90
.LBB0_169:
	incq	%r9
	addq	$232, %r8
.LBB0_165:
	cmpq	$57, %r9
	jg	.LBB0_170
	xorl	%r10d, %r10d
	cmpq	$57, %r10
	jg	.LBB0_169
	.p2align	4, 0x90
.LBB0_168:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$57, %r10
	jle	.LBB0_168
	jmp	.LBB0_169
.LBB0_172:
	movq	%rsp, %r15
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$56, %r8d
	movq	%r8, -56(%rdx)
	movl	$64, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-96(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$200704, %edi
	movq	%rdi, -40(%rdx)
	movl	$3136, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$58, -24(%rdx)
	movq	$3364, -32(%rdx)
	movq	$215296, -40(%rdx)
	movq	$56, -48(%rdx)
	movq	$56, -56(%rdx)
	movq	$64, -64(%rdx)
	movq	$59, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r15, %rsp
	movq	%rbx, %rcx
	movq	6176(%rbp), %r10
	jmp	.LBB0_173
	.p2align	4, 0x90
.LBB0_183:
	incq	%rax
	addq	$802816, %rcx
.LBB0_173:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_184
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_175
	.p2align	4, 0x90
.LBB0_182:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_175:
	cmpq	$63, %rsi
	jg	.LBB0_183
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_177
	.p2align	4, 0x90
.LBB0_181:
	incq	%r8
	addq	$224, %rdi
.LBB0_177:
	cmpq	$55, %r8
	jg	.LBB0_182
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_181
	.p2align	4, 0x90
.LBB0_180:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_180
	jmp	.LBB0_181
.LBB0_184:
	xorl	%eax, %eax
	jmp	.LBB0_185
	.p2align	4, 0x90
.LBB0_204:
	incq	%rax
	addq	$861184, -72(%rbp)
.LBB0_185:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_205
	movq	6256(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_187
	.p2align	4, 0x90
.LBB0_203:
	incq	%rdx
	addq	$2304, -56(%rbp)
.LBB0_187:
	cmpq	$63, %rdx
	jg	.LBB0_204
	movq	-72(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_189
	.p2align	4, 0x90
.LBB0_202:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$232, %rcx
.LBB0_189:
	cmpq	$55, %rdi
	jg	.LBB0_203
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_191
	.p2align	4, 0x90
.LBB0_201:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_191:
	cmpq	$55, %r9
	jg	.LBB0_202
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r12
	movq	-56(%rbp), %rcx
	xorl	%r15d, %r15d
	jmp	.LBB0_193
	.p2align	4, 0x90
.LBB0_200:
	incq	%r15
	addq	$36, %rcx
	movq	-88(%rbp), %r12
	addq	$13456, %r12
.LBB0_193:
	cmpq	$63, %r15
	jg	.LBB0_201
	movq	%r12, -88(%rbp)
	movq	%rcx, %r11
	xorl	%esi, %esi
	jmp	.LBB0_195
	.p2align	4, 0x90
.LBB0_199:
	incq	%rsi
	addq	$12, %r11
	addq	$232, %r12
.LBB0_195:
	cmpq	$2, %rsi
	jg	.LBB0_200
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_199
	.p2align	4, 0x90
.LBB0_198:
	movss	(%r12,%r14,4), %xmm0
	imulq	$200704, %rax, %r8
	imulq	$3136, %rdx, %r10
	addq	%r8, %r10
	imulq	$56, %rdi, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%r8,4), %xmm0
	movss	%xmm0, (%rbx,%r8,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_198
	jmp	.LBB0_199
.LBB0_205:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_206
	.p2align	4, 0x90
.LBB0_216:
	incq	%rax
	addq	$802816, %rcx
.LBB0_206:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_217
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_208
	.p2align	4, 0x90
.LBB0_215:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_208:
	cmpq	$63, %rsi
	jg	.LBB0_216
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_210
	.p2align	4, 0x90
.LBB0_214:
	incq	%r8
	addq	$224, %rdi
.LBB0_210:
	cmpq	$55, %r8
	jg	.LBB0_215
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_214
	.p2align	4, 0x90
.LBB0_213:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_213
	jmp	.LBB0_214
.LBB0_217:
	movq	-104(%rbp), %r14
	movq	%r14, %rdi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	movq	%r14, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	xorl	%eax, %eax
	movq	%r12, %rcx
	movq	5920(%rbp), %r10
	jmp	.LBB0_218
	.p2align	4, 0x90
.LBB0_228:
	incq	%rax
	addq	$3211264, %rcx
.LBB0_218:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_229
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_220
	.p2align	4, 0x90
.LBB0_227:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_220:
	cmpq	$255, %rsi
	jg	.LBB0_228
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_222
	.p2align	4, 0x90
.LBB0_226:
	incq	%r8
	addq	$224, %rdi
.LBB0_222:
	cmpq	$55, %r8
	jg	.LBB0_227
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_226
	.p2align	4, 0x90
.LBB0_225:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_225
	jmp	.LBB0_226
.LBB0_229:
	xorl	%eax, %eax
	jmp	.LBB0_230
	.p2align	4, 0x90
.LBB0_249:
	incq	%rax
	addq	$802816, %rbx
.LBB0_230:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_250
	movq	6000(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_232
	.p2align	4, 0x90
.LBB0_248:
	incq	%rdx
	addq	$256, -72(%rbp)
.LBB0_232:
	cmpq	$255, %rdx
	jg	.LBB0_249
	movq	%rbx, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_234
	.p2align	4, 0x90
.LBB0_247:
	incq	%rdi
	movq	-96(%rbp), %rcx
	addq	$224, %rcx
.LBB0_234:
	cmpq	$55, %rdi
	jg	.LBB0_248
	movq	%rcx, -96(%rbp)
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_236
	.p2align	4, 0x90
.LBB0_246:
	incq	%r9
	addq	$4, -64(%rbp)
.LBB0_236:
	cmpq	$55, %r9
	jg	.LBB0_247
	movq	-64(%rbp), %r10
	movq	-72(%rbp), %r11
	xorl	%ecx, %ecx
	jmp	.LBB0_238
	.p2align	4, 0x90
.LBB0_245:
	movq	-56(%rbp), %rcx
	incq	%rcx
	movq	-80(%rbp), %r11
	addq	$4, %r11
	movq	-88(%rbp), %r10
	addq	$12544, %r10
.LBB0_238:
	cmpq	$63, %rcx
	jg	.LBB0_246
	movq	%rcx, -56(%rbp)
	movq	%r10, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_240
	.p2align	4, 0x90
.LBB0_244:
	incq	%rsi
	addq	$4, %r11
	addq	$224, %r10
.LBB0_240:
	testq	%rsi, %rsi
	jg	.LBB0_245
	xorl	%r8d, %r8d
	testq	%r8, %r8
	jg	.LBB0_244
	.p2align	4, 0x90
.LBB0_243:
	movss	(%r10,%r8,4), %xmm0
	imulq	$802816, %rax, %rcx
	imulq	$3136, %rdx, %r14
	addq	%rcx, %r14
	imulq	$56, %rdi, %rcx
	addq	%r9, %rcx
	addq	%r14, %rcx
	mulss	(%r11,%r8,4), %xmm0
	addss	(%r12,%rcx,4), %xmm0
	movss	%xmm0, (%r12,%rcx,4)
	incq	%r8
	testq	%r8, %r8
	jle	.LBB0_243
	jmp	.LBB0_244
.LBB0_250:
	xorl	%eax, %eax
	movq	%r15, %rcx
	movq	6688(%rbp), %r10
	jmp	.LBB0_251
	.p2align	4, 0x90
.LBB0_261:
	incq	%rax
	addq	$3211264, %rcx
.LBB0_251:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_262
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_253
	.p2align	4, 0x90
.LBB0_260:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_253:
	cmpq	$255, %rsi
	jg	.LBB0_261
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_255
	.p2align	4, 0x90
.LBB0_259:
	incq	%r8
	addq	$224, %rdi
.LBB0_255:
	cmpq	$55, %r8
	jg	.LBB0_260
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_259
	.p2align	4, 0x90
.LBB0_258:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_258
	jmp	.LBB0_259
.LBB0_262:
	xorl	%eax, %eax
	jmp	.LBB0_263
	.p2align	4, 0x90
.LBB0_282:
	incq	%rax
	addq	$802816, %r13
.LBB0_263:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_283
	movq	6768(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_265
	.p2align	4, 0x90
.LBB0_281:
	incq	%rdx
	addq	$256, -64(%rbp)
.LBB0_265:
	cmpq	$255, %rdx
	jg	.LBB0_282
	movq	%r13, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_267
	.p2align	4, 0x90
.LBB0_280:
	incq	%rdi
	movq	-72(%rbp), %rcx
	addq	$224, %rcx
.LBB0_267:
	cmpq	$55, %rdi
	jg	.LBB0_281
	movq	%rcx, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_269
	.p2align	4, 0x90
.LBB0_279:
	incq	%r9
	movq	-56(%rbp), %rcx
	addq	$4, %rcx
.LBB0_269:
	cmpq	$55, %r9
	jg	.LBB0_280
	movq	%rcx, -56(%rbp)
	movq	%rcx, %r14
	movq	-64(%rbp), %r11
	xorl	%ebx, %ebx
	jmp	.LBB0_271
	.p2align	4, 0x90
.LBB0_278:
	incq	%rbx
	movq	-80(%rbp), %r11
	addq	$4, %r11
	movq	-88(%rbp), %r14
	addq	$12544, %r14
.LBB0_271:
	cmpq	$63, %rbx
	jg	.LBB0_279
	movq	%r14, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_273
	.p2align	4, 0x90
.LBB0_277:
	incq	%rsi
	addq	$4, %r11
	addq	$224, %r14
.LBB0_273:
	testq	%rsi, %rsi
	jg	.LBB0_278
	xorl	%r8d, %r8d
	testq	%r8, %r8
	jg	.LBB0_277
	.p2align	4, 0x90
.LBB0_276:
	movss	(%r14,%r8,4), %xmm0
	imulq	$802816, %rax, %r10
	imulq	$3136, %rdx, %rcx
	addq	%r10, %rcx
	imulq	$56, %rdi, %r10
	addq	%r9, %r10
	addq	%rcx, %r10
	mulss	(%r11,%r8,4), %xmm0
	addss	(%r15,%r10,4), %xmm0
	movss	%xmm0, (%r15,%r10,4)
	incq	%r8
	testq	%r8, %r8
	jle	.LBB0_276
	jmp	.LBB0_277
.LBB0_283:
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_284
	.p2align	4, 0x90
.LBB0_294:
	incq	%rax
	addq	$3211264, %r15
	addq	$3211264, %rcx
.LBB0_284:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_295
	movq	%rcx, %rdx
	movq	%r15, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_286
	.p2align	4, 0x90
.LBB0_293:
	incq	%rdi
	addq	$12544, %rsi
	addq	$12544, %rdx
.LBB0_286:
	cmpq	$255, %rdi
	jg	.LBB0_294
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_288
	.p2align	4, 0x90
.LBB0_292:
	incq	%r10
	addq	$224, %r9
	addq	$224, %r8
.LBB0_288:
	cmpq	$55, %r10
	jg	.LBB0_293
	xorl	%r11d, %r11d
	cmpq	$55, %r11
	jg	.LBB0_292
	.p2align	4, 0x90
.LBB0_291:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$55, %r11
	jle	.LBB0_291
	jmp	.LBB0_292
.LBB0_295:
	movq	-104(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_296
	.p2align	4, 0x90
.LBB0_306:
	incq	%rbx
	addq	$3211264, %rcx
	addq	$3211264, %r12
.LBB0_296:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_307
	movq	%r12, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_298
	.p2align	4, 0x90
.LBB0_305:
	incq	%rdi
	addq	$12544, %rsi
	addq	$12544, %rdx
.LBB0_298:
	cmpq	$255, %rdi
	jg	.LBB0_306
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_300
	.p2align	4, 0x90
.LBB0_304:
	incq	%r10
	addq	$224, %r9
	addq	$224, %r8
.LBB0_300:
	cmpq	$55, %r10
	jg	.LBB0_305
	xorl	%r11d, %r11d
	cmpq	$55, %r11
	jg	.LBB0_304
	.p2align	4, 0x90
.LBB0_303:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$55, %r11
	jle	.LBB0_303
	jmp	.LBB0_304
.LBB0_307:
	movq	%rax, -96(%rbp)
	movq	-128(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -112(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	5664(%rbp), %r10
	jmp	.LBB0_308
	.p2align	4, 0x90
.LBB0_318:
	incq	%rax
	addq	$802816, %rcx
.LBB0_308:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_319
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_310
	.p2align	4, 0x90
.LBB0_317:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_310:
	cmpq	$63, %rsi
	jg	.LBB0_318
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_312
	.p2align	4, 0x90
.LBB0_316:
	incq	%r8
	addq	$224, %rdi
.LBB0_312:
	cmpq	$55, %r8
	jg	.LBB0_317
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_316
	.p2align	4, 0x90
.LBB0_315:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_315
	jmp	.LBB0_316
.LBB0_319:
	xorl	%eax, %eax
	movq	-96(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	jmp	.LBB0_320
	.p2align	4, 0x90
.LBB0_339:
	incq	%rax
	addq	$3211264, -72(%rbp)
.LBB0_320:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_340
	movq	5744(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_322
	.p2align	4, 0x90
.LBB0_338:
	incq	%rsi
	addq	$1024, -56(%rbp)
.LBB0_322:
	cmpq	$63, %rsi
	jg	.LBB0_339
	movq	-72(%rbp), %r11
	xorl	%r8d, %r8d
	jmp	.LBB0_324
	.p2align	4, 0x90
.LBB0_337:
	incq	%r8
	movq	-64(%rbp), %r11
	addq	$224, %r11
.LBB0_324:
	cmpq	$55, %r8
	jg	.LBB0_338
	movq	%r11, -64(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_326
	.p2align	4, 0x90
.LBB0_336:
	incq	%r10
	movq	-80(%rbp), %r11
	addq	$4, %r11
.LBB0_326:
	cmpq	$55, %r10
	jg	.LBB0_337
	movq	%r11, -80(%rbp)
	movq	-56(%rbp), %rdx
	xorl	%r12d, %r12d
	jmp	.LBB0_328
	.p2align	4, 0x90
.LBB0_335:
	incq	%r12
	addq	$4, %rdx
	movq	-88(%rbp), %r11
	addq	$12544, %r11
.LBB0_328:
	cmpq	$255, %r12
	jg	.LBB0_336
	movq	%r11, -88(%rbp)
	movq	%rdx, %r14
	xorl	%ecx, %ecx
	jmp	.LBB0_330
	.p2align	4, 0x90
.LBB0_334:
	incq	%rcx
	addq	$4, %r14
	addq	$224, %r11
.LBB0_330:
	testq	%rcx, %rcx
	jg	.LBB0_335
	xorl	%r13d, %r13d
	testq	%r13, %r13
	jg	.LBB0_334
	.p2align	4, 0x90
.LBB0_333:
	movss	(%r11,%r13,4), %xmm0
	imulq	$200704, %rax, %rdi
	imulq	$3136, %rsi, %r9
	addq	%rdi, %r9
	imulq	$56, %r8, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r14,%r13,4), %xmm0
	addss	(%rbx,%rdi,4), %xmm0
	movss	%xmm0, (%rbx,%rdi,4)
	incq	%r13
	testq	%r13, %r13
	jle	.LBB0_333
	jmp	.LBB0_334
.LBB0_340:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_341
	.p2align	4, 0x90
.LBB0_351:
	incq	%rax
	addq	$802816, %rcx
.LBB0_341:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_352
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_343
	.p2align	4, 0x90
.LBB0_350:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_343:
	cmpq	$63, %rsi
	jg	.LBB0_351
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_345
	.p2align	4, 0x90
.LBB0_349:
	incq	%r8
	addq	$224, %rdi
.LBB0_345:
	cmpq	$55, %r8
	jg	.LBB0_350
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_349
	.p2align	4, 0x90
.LBB0_348:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_348
	jmp	.LBB0_349
.LBB0_352:
	movq	-136(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -72(%rbp)
	jmp	.LBB0_353
	.p2align	4, 0x90
.LBB0_363:
	incq	%rcx
	addq	$861184, %rdx
.LBB0_353:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_364
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_355
	.p2align	4, 0x90
.LBB0_362:
	incq	%rdi
	addq	$13456, %rsi
.LBB0_355:
	cmpq	$63, %rdi
	jg	.LBB0_363
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_357
	.p2align	4, 0x90
.LBB0_361:
	incq	%r9
	addq	$232, %r8
.LBB0_357:
	cmpq	$57, %r9
	jg	.LBB0_362
	xorl	%r10d, %r10d
	cmpq	$57, %r10
	jg	.LBB0_361
	.p2align	4, 0x90
.LBB0_360:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$57, %r10
	jle	.LBB0_360
	jmp	.LBB0_361
.LBB0_364:
	movq	%rsp, %r12
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$56, %r8d
	movq	%r8, -56(%rdx)
	movl	$64, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-112(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$200704, %edi
	movq	%rdi, -40(%rdx)
	movl	$3136, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$58, -24(%rdx)
	movq	$3364, -32(%rdx)
	movq	$215296, -40(%rdx)
	movq	$56, -48(%rdx)
	movq	$56, -56(%rdx)
	movq	$64, -64(%rdx)
	movq	$59, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r12, %rsp
	movq	%rbx, %rcx
	movq	5408(%rbp), %r10
	jmp	.LBB0_365
	.p2align	4, 0x90
.LBB0_375:
	incq	%rax
	addq	$802816, %rcx
.LBB0_365:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_376
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_367
	.p2align	4, 0x90
.LBB0_374:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_367:
	cmpq	$63, %rsi
	jg	.LBB0_375
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_369
	.p2align	4, 0x90
.LBB0_373:
	incq	%r8
	addq	$224, %rdi
.LBB0_369:
	cmpq	$55, %r8
	jg	.LBB0_374
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_373
	.p2align	4, 0x90
.LBB0_372:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_372
	jmp	.LBB0_373
.LBB0_376:
	xorl	%eax, %eax
	jmp	.LBB0_377
	.p2align	4, 0x90
.LBB0_396:
	incq	%rax
	addq	$861184, -72(%rbp)
.LBB0_377:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_397
	movq	5488(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_379
	.p2align	4, 0x90
.LBB0_395:
	incq	%rdx
	addq	$2304, -56(%rbp)
.LBB0_379:
	cmpq	$63, %rdx
	jg	.LBB0_396
	movq	-72(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_381
	.p2align	4, 0x90
.LBB0_394:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$232, %rcx
.LBB0_381:
	cmpq	$55, %rdi
	jg	.LBB0_395
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_383
	.p2align	4, 0x90
.LBB0_393:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_383:
	cmpq	$55, %r9
	jg	.LBB0_394
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r13
	movq	-56(%rbp), %rcx
	xorl	%r12d, %r12d
	jmp	.LBB0_385
	.p2align	4, 0x90
.LBB0_392:
	incq	%r12
	addq	$36, %rcx
	movq	-88(%rbp), %r13
	addq	$13456, %r13
.LBB0_385:
	cmpq	$63, %r12
	jg	.LBB0_393
	movq	%r13, -88(%rbp)
	movq	%rcx, %r11
	xorl	%esi, %esi
	jmp	.LBB0_387
	.p2align	4, 0x90
.LBB0_391:
	incq	%rsi
	addq	$12, %r11
	addq	$232, %r13
.LBB0_387:
	cmpq	$2, %rsi
	jg	.LBB0_392
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_391
	.p2align	4, 0x90
.LBB0_390:
	movss	(%r13,%r14,4), %xmm0
	imulq	$200704, %rax, %r8
	imulq	$3136, %rdx, %r10
	addq	%r8, %r10
	imulq	$56, %rdi, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%r8,4), %xmm0
	movss	%xmm0, (%rbx,%r8,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_390
	jmp	.LBB0_391
.LBB0_397:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_398
	.p2align	4, 0x90
.LBB0_408:
	incq	%rax
	addq	$802816, %rcx
.LBB0_398:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_409
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_400
	.p2align	4, 0x90
.LBB0_407:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_400:
	cmpq	$63, %rsi
	jg	.LBB0_408
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_402
	.p2align	4, 0x90
.LBB0_406:
	incq	%r8
	addq	$224, %rdi
.LBB0_402:
	cmpq	$55, %r8
	jg	.LBB0_407
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_406
	.p2align	4, 0x90
.LBB0_405:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_405
	jmp	.LBB0_406
.LBB0_409:
	xorl	%eax, %eax
	movq	%r15, %rcx
	jmp	.LBB0_410
	.p2align	4, 0x90
.LBB0_420:
	incq	%rax
	addq	$3211264, %rcx
.LBB0_410:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_421
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_412
	.p2align	4, 0x90
.LBB0_419:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_412:
	cmpq	$255, %rsi
	jg	.LBB0_420
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_414
	.p2align	4, 0x90
.LBB0_418:
	incq	%r8
	addq	$224, %rdi
.LBB0_414:
	cmpq	$55, %r8
	jg	.LBB0_419
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_418
	.p2align	4, 0x90
.LBB0_417:
	movq	5152(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_417
	jmp	.LBB0_418
.LBB0_421:
	xorl	%eax, %eax
	jmp	.LBB0_422
	.p2align	4, 0x90
.LBB0_441:
	incq	%rax
	addq	$802816, %rbx
.LBB0_422:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_442
	movq	5232(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_424
	.p2align	4, 0x90
.LBB0_440:
	incq	%rdx
	addq	$256, -56(%rbp)
.LBB0_424:
	cmpq	$255, %rdx
	jg	.LBB0_441
	movq	%rbx, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_426
	.p2align	4, 0x90
.LBB0_439:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$224, %rcx
.LBB0_426:
	cmpq	$55, %rdi
	jg	.LBB0_440
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_428
	.p2align	4, 0x90
.LBB0_438:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_428:
	cmpq	$55, %r9
	jg	.LBB0_439
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r12
	movq	-56(%rbp), %r11
	xorl	%r14d, %r14d
	jmp	.LBB0_430
	.p2align	4, 0x90
.LBB0_437:
	incq	%r14
	addq	$4, %r11
	movq	-88(%rbp), %r12
	addq	$12544, %r12
.LBB0_430:
	cmpq	$63, %r14
	jg	.LBB0_438
	movq	%r12, -88(%rbp)
	movq	%r11, %r13
	xorl	%esi, %esi
	jmp	.LBB0_432
	.p2align	4, 0x90
.LBB0_436:
	incq	%rsi
	addq	$4, %r13
	addq	$224, %r12
.LBB0_432:
	testq	%rsi, %rsi
	jg	.LBB0_437
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_436
	.p2align	4, 0x90
.LBB0_435:
	movss	(%r12,%rcx,4), %xmm0
	imulq	$802816, %rax, %r8
	imulq	$3136, %rdx, %r10
	addq	%r8, %r10
	imulq	$56, %rdi, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r15,%r8,4), %xmm0
	movss	%xmm0, (%r15,%r8,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_435
	jmp	.LBB0_436
.LBB0_442:
	xorl	%eax, %eax
	movq	%r15, %rcx
	movq	-96(%rbp), %rbx
	jmp	.LBB0_443
	.p2align	4, 0x90
.LBB0_453:
	incq	%rax
	addq	$3211264, %rbx
	addq	$3211264, %rcx
.LBB0_443:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_454
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_445
	.p2align	4, 0x90
.LBB0_452:
	incq	%rdi
	addq	$12544, %rsi
	addq	$12544, %rdx
.LBB0_445:
	cmpq	$255, %rdi
	jg	.LBB0_453
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_447
	.p2align	4, 0x90
.LBB0_451:
	incq	%r10
	addq	$224, %r9
	addq	$224, %r8
.LBB0_447:
	cmpq	$55, %r10
	jg	.LBB0_452
	xorl	%r11d, %r11d
	cmpq	$55, %r11
	jg	.LBB0_451
	.p2align	4, 0x90
.LBB0_450:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$55, %r11
	jle	.LBB0_450
	jmp	.LBB0_451
.LBB0_454:
	movq	-104(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r13
	addq	$63, %r13
	andq	$-64, %r13
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_455
	.p2align	4, 0x90
.LBB0_465:
	incq	%rbx
	addq	$3211264, %rcx
	addq	$3211264, %r15
.LBB0_455:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_466
	movq	%r15, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_457
	.p2align	4, 0x90
.LBB0_464:
	incq	%rdi
	addq	$12544, %rsi
	addq	$12544, %rdx
.LBB0_457:
	cmpq	$255, %rdi
	jg	.LBB0_465
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_459
	.p2align	4, 0x90
.LBB0_463:
	incq	%r10
	addq	$224, %r9
	addq	$224, %r8
.LBB0_459:
	cmpq	$55, %r10
	jg	.LBB0_464
	xorl	%r11d, %r11d
	cmpq	$55, %r11
	jg	.LBB0_463
	.p2align	4, 0x90
.LBB0_462:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$55, %r11
	jle	.LBB0_462
	jmp	.LBB0_463
.LBB0_466:
	movq	%rax, -96(%rbp)
	movq	-128(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -112(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	5024(%rbp), %r10
	jmp	.LBB0_467
	.p2align	4, 0x90
.LBB0_477:
	incq	%rax
	addq	$802816, %rcx
.LBB0_467:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_478
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_469
	.p2align	4, 0x90
.LBB0_476:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_469:
	cmpq	$63, %rsi
	jg	.LBB0_477
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_471
	.p2align	4, 0x90
.LBB0_475:
	incq	%r8
	addq	$224, %rdi
.LBB0_471:
	cmpq	$55, %r8
	jg	.LBB0_476
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_475
	.p2align	4, 0x90
.LBB0_474:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_474
	jmp	.LBB0_475
.LBB0_478:
	xorl	%eax, %eax
	movq	-96(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	jmp	.LBB0_479
	.p2align	4, 0x90
.LBB0_498:
	incq	%rax
	addq	$3211264, -72(%rbp)
.LBB0_479:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_499
	movq	5064(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_481
	.p2align	4, 0x90
.LBB0_497:
	incq	%rsi
	addq	$1024, -56(%rbp)
.LBB0_481:
	cmpq	$63, %rsi
	jg	.LBB0_498
	movq	-72(%rbp), %r11
	xorl	%r8d, %r8d
	jmp	.LBB0_483
	.p2align	4, 0x90
.LBB0_496:
	incq	%r8
	movq	-64(%rbp), %r11
	addq	$224, %r11
.LBB0_483:
	cmpq	$55, %r8
	jg	.LBB0_497
	movq	%r11, -64(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_485
	.p2align	4, 0x90
.LBB0_495:
	incq	%r10
	movq	-80(%rbp), %r11
	addq	$4, %r11
.LBB0_485:
	cmpq	$55, %r10
	jg	.LBB0_496
	movq	%r11, -80(%rbp)
	movq	-56(%rbp), %rdx
	xorl	%r15d, %r15d
	jmp	.LBB0_487
	.p2align	4, 0x90
.LBB0_494:
	incq	%r15
	addq	$4, %rdx
	movq	-88(%rbp), %r11
	addq	$12544, %r11
.LBB0_487:
	cmpq	$255, %r15
	jg	.LBB0_495
	movq	%r11, -88(%rbp)
	movq	%rdx, %r14
	xorl	%ecx, %ecx
	jmp	.LBB0_489
	.p2align	4, 0x90
.LBB0_493:
	incq	%rcx
	addq	$4, %r14
	addq	$224, %r11
.LBB0_489:
	testq	%rcx, %rcx
	jg	.LBB0_494
	xorl	%r12d, %r12d
	testq	%r12, %r12
	jg	.LBB0_493
	.p2align	4, 0x90
.LBB0_492:
	movss	(%r11,%r12,4), %xmm0
	imulq	$200704, %rax, %rdi
	imulq	$3136, %rsi, %r9
	addq	%rdi, %r9
	imulq	$56, %r8, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r14,%r12,4), %xmm0
	addss	(%rbx,%rdi,4), %xmm0
	movss	%xmm0, (%rbx,%rdi,4)
	incq	%r12
	testq	%r12, %r12
	jle	.LBB0_492
	jmp	.LBB0_493
.LBB0_499:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_500
	.p2align	4, 0x90
.LBB0_510:
	incq	%rax
	addq	$802816, %rcx
.LBB0_500:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_511
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_502
	.p2align	4, 0x90
.LBB0_509:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_502:
	cmpq	$63, %rsi
	jg	.LBB0_510
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_504
	.p2align	4, 0x90
.LBB0_508:
	incq	%r8
	addq	$224, %rdi
.LBB0_504:
	cmpq	$55, %r8
	jg	.LBB0_509
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_508
	.p2align	4, 0x90
.LBB0_507:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_507
	jmp	.LBB0_508
.LBB0_511:
	movq	-136(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -72(%rbp)
	jmp	.LBB0_512
	.p2align	4, 0x90
.LBB0_522:
	incq	%rcx
	addq	$861184, %rdx
.LBB0_512:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_523
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_514
	.p2align	4, 0x90
.LBB0_521:
	incq	%rdi
	addq	$13456, %rsi
.LBB0_514:
	cmpq	$63, %rdi
	jg	.LBB0_522
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_516
	.p2align	4, 0x90
.LBB0_520:
	incq	%r9
	addq	$232, %r8
.LBB0_516:
	cmpq	$57, %r9
	jg	.LBB0_521
	xorl	%r10d, %r10d
	cmpq	$57, %r10
	jg	.LBB0_520
	.p2align	4, 0x90
.LBB0_519:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$57, %r10
	jle	.LBB0_519
	jmp	.LBB0_520
.LBB0_523:
	movq	%rsp, %r15
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$56, %r8d
	movq	%r8, -56(%rdx)
	movl	$64, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-112(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$200704, %edi
	movq	%rdi, -40(%rdx)
	movl	$3136, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$58, -24(%rdx)
	movq	$3364, -32(%rdx)
	movq	$215296, -40(%rdx)
	movq	$56, -48(%rdx)
	movq	$56, -56(%rdx)
	movq	$64, -64(%rdx)
	movq	$59, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r15, %rsp
	movq	%rbx, %rcx
	movq	4856(%rbp), %r10
	jmp	.LBB0_524
	.p2align	4, 0x90
.LBB0_534:
	incq	%rax
	addq	$802816, %rcx
.LBB0_524:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_535
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_526
	.p2align	4, 0x90
.LBB0_533:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_526:
	cmpq	$63, %rsi
	jg	.LBB0_534
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_528
	.p2align	4, 0x90
.LBB0_532:
	incq	%r8
	addq	$224, %rdi
.LBB0_528:
	cmpq	$55, %r8
	jg	.LBB0_533
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_532
	.p2align	4, 0x90
.LBB0_531:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_531
	jmp	.LBB0_532
.LBB0_535:
	xorl	%eax, %eax
	jmp	.LBB0_536
	.p2align	4, 0x90
.LBB0_555:
	incq	%rax
	addq	$861184, -72(%rbp)
.LBB0_536:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_556
	movq	4936(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_538
	.p2align	4, 0x90
.LBB0_554:
	incq	%rdx
	addq	$2304, -56(%rbp)
.LBB0_538:
	cmpq	$63, %rdx
	jg	.LBB0_555
	movq	-72(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_540
	.p2align	4, 0x90
.LBB0_553:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$232, %rcx
.LBB0_540:
	cmpq	$55, %rdi
	jg	.LBB0_554
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_542
	.p2align	4, 0x90
.LBB0_552:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_542:
	cmpq	$55, %r9
	jg	.LBB0_553
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r12
	movq	-56(%rbp), %rcx
	xorl	%r15d, %r15d
	jmp	.LBB0_544
	.p2align	4, 0x90
.LBB0_551:
	incq	%r15
	addq	$36, %rcx
	movq	-88(%rbp), %r12
	addq	$13456, %r12
.LBB0_544:
	cmpq	$63, %r15
	jg	.LBB0_552
	movq	%r12, -88(%rbp)
	movq	%rcx, %r11
	xorl	%esi, %esi
	jmp	.LBB0_546
	.p2align	4, 0x90
.LBB0_550:
	incq	%rsi
	addq	$12, %r11
	addq	$232, %r12
.LBB0_546:
	cmpq	$2, %rsi
	jg	.LBB0_551
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_550
	.p2align	4, 0x90
.LBB0_549:
	movss	(%r12,%r14,4), %xmm0
	imulq	$200704, %rax, %r8
	imulq	$3136, %rdx, %r10
	addq	%r8, %r10
	imulq	$56, %rdi, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%r8,4), %xmm0
	movss	%xmm0, (%rbx,%r8,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_549
	jmp	.LBB0_550
.LBB0_556:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_557
	.p2align	4, 0x90
.LBB0_567:
	incq	%rax
	addq	$802816, %rcx
.LBB0_557:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_568
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_559
	.p2align	4, 0x90
.LBB0_566:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_559:
	cmpq	$63, %rsi
	jg	.LBB0_567
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_561
	.p2align	4, 0x90
.LBB0_565:
	incq	%r8
	addq	$224, %rdi
.LBB0_561:
	cmpq	$55, %r8
	jg	.LBB0_566
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_565
	.p2align	4, 0x90
.LBB0_564:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_564
	jmp	.LBB0_565
.LBB0_568:
	xorl	%eax, %eax
	movq	%r13, %rcx
	jmp	.LBB0_569
	.p2align	4, 0x90
.LBB0_579:
	incq	%rax
	addq	$3211264, %rcx
.LBB0_569:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_580
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_571
	.p2align	4, 0x90
.LBB0_578:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_571:
	cmpq	$255, %rsi
	jg	.LBB0_579
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_573
	.p2align	4, 0x90
.LBB0_577:
	incq	%r8
	addq	$224, %rdi
.LBB0_573:
	cmpq	$55, %r8
	jg	.LBB0_578
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_577
	.p2align	4, 0x90
.LBB0_576:
	movq	4648(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_576
	jmp	.LBB0_577
.LBB0_580:
	xorl	%eax, %eax
	jmp	.LBB0_581
	.p2align	4, 0x90
.LBB0_600:
	incq	%rax
	addq	$802816, %rbx
.LBB0_581:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_601
	movq	4728(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_583
	.p2align	4, 0x90
.LBB0_599:
	incq	%rdx
	addq	$256, -56(%rbp)
.LBB0_583:
	cmpq	$255, %rdx
	jg	.LBB0_600
	movq	%rbx, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_585
	.p2align	4, 0x90
.LBB0_598:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$224, %rcx
.LBB0_585:
	cmpq	$55, %rdi
	jg	.LBB0_599
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_587
	.p2align	4, 0x90
.LBB0_597:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_587:
	cmpq	$55, %r9
	jg	.LBB0_598
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r15
	movq	-56(%rbp), %r11
	xorl	%r14d, %r14d
	jmp	.LBB0_589
	.p2align	4, 0x90
.LBB0_596:
	incq	%r14
	addq	$4, %r11
	movq	-88(%rbp), %r15
	addq	$12544, %r15
.LBB0_589:
	cmpq	$63, %r14
	jg	.LBB0_597
	movq	%r15, -88(%rbp)
	movq	%r11, %r12
	xorl	%esi, %esi
	jmp	.LBB0_591
	.p2align	4, 0x90
.LBB0_595:
	incq	%rsi
	addq	$4, %r12
	addq	$224, %r15
.LBB0_591:
	testq	%rsi, %rsi
	jg	.LBB0_596
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_595
	.p2align	4, 0x90
.LBB0_594:
	movss	(%r15,%rcx,4), %xmm0
	imulq	$802816, %rax, %r8
	imulq	$3136, %rdx, %r10
	addq	%r8, %r10
	imulq	$56, %rdi, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r12,%rcx,4), %xmm0
	addss	(%r13,%r8,4), %xmm0
	movss	%xmm0, (%r13,%r8,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_594
	jmp	.LBB0_595
.LBB0_601:
	xorl	%eax, %eax
	movq	%r13, %rcx
	movq	-96(%rbp), %rbx
	jmp	.LBB0_602
	.p2align	4, 0x90
.LBB0_612:
	incq	%rax
	addq	$3211264, %rbx
	addq	$3211264, %rcx
.LBB0_602:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_613
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_604
	.p2align	4, 0x90
.LBB0_611:
	incq	%rdi
	addq	$12544, %rsi
	addq	$12544, %rdx
.LBB0_604:
	cmpq	$255, %rdi
	jg	.LBB0_612
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_606
	.p2align	4, 0x90
.LBB0_610:
	incq	%r10
	addq	$224, %r9
	addq	$224, %r8
.LBB0_606:
	cmpq	$55, %r10
	jg	.LBB0_611
	xorl	%r11d, %r11d
	cmpq	$55, %r11
	jg	.LBB0_610
	.p2align	4, 0x90
.LBB0_609:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$55, %r11
	jle	.LBB0_609
	jmp	.LBB0_610
.LBB0_613:
	movq	-104(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rcx
	addq	$63, %rcx
	andq	$-64, %rcx
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rcx, -96(%rbp)
	jmp	.LBB0_614
	.p2align	4, 0x90
.LBB0_624:
	incq	%rax
	addq	$3211264, %rcx
	addq	$3211264, %r13
.LBB0_614:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_625
	movq	%r13, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_616
	.p2align	4, 0x90
.LBB0_623:
	incq	%rdi
	addq	$12544, %rsi
	addq	$12544, %rdx
.LBB0_616:
	cmpq	$255, %rdi
	jg	.LBB0_624
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_618
	.p2align	4, 0x90
.LBB0_622:
	incq	%r10
	addq	$224, %r9
	addq	$224, %r8
.LBB0_618:
	cmpq	$55, %r10
	jg	.LBB0_623
	xorl	%r11d, %r11d
	cmpq	$55, %r11
	jg	.LBB0_622
	.p2align	4, 0x90
.LBB0_621:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$55, %r11
	jle	.LBB0_621
	jmp	.LBB0_622
.LBB0_625:
	imulq	$1605632, -48(%rbp), %rdi
	orq	$64, %rdi
	movq	%rdi, -104(%rbp)
	callq	malloc@PLT
	movq	%rax, -72(%rbp)
	leaq	63(%rax), %r14
	andq	$-64, %r14
	xorl	%eax, %eax
	movq	%r14, %rcx
	movq	2904(%rbp), %r10
	jmp	.LBB0_626
	.p2align	4, 0x90
.LBB0_636:
	incq	%rax
	addq	$1605632, %rcx
.LBB0_626:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_637
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_628
	.p2align	4, 0x90
.LBB0_635:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_628:
	cmpq	$127, %rsi
	jg	.LBB0_636
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_630
	.p2align	4, 0x90
.LBB0_634:
	incq	%r8
	addq	$224, %rdi
.LBB0_630:
	cmpq	$55, %r8
	jg	.LBB0_635
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_634
	.p2align	4, 0x90
.LBB0_633:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_633
	jmp	.LBB0_634
.LBB0_637:
	xorl	%eax, %eax
	movq	-96(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	jmp	.LBB0_638
	.p2align	4, 0x90
.LBB0_657:
	incq	%rax
	addq	$3211264, -64(%rbp)
.LBB0_638:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_658
	movq	2944(%rbp), %rcx
	movq	%rcx, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_640
	.p2align	4, 0x90
.LBB0_656:
	incq	%rsi
	addq	$1024, -80(%rbp)
.LBB0_640:
	cmpq	$127, %rsi
	jg	.LBB0_657
	movq	-64(%rbp), %r9
	xorl	%r8d, %r8d
	jmp	.LBB0_642
	.p2align	4, 0x90
.LBB0_655:
	incq	%r8
	movq	-56(%rbp), %r9
	addq	$224, %r9
.LBB0_642:
	cmpq	$55, %r8
	jg	.LBB0_656
	movq	%r9, -56(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_644
	.p2align	4, 0x90
.LBB0_654:
	incq	%r10
	movq	-88(%rbp), %r9
	addq	$4, %r9
.LBB0_644:
	cmpq	$55, %r10
	jg	.LBB0_655
	movq	%r9, -88(%rbp)
	movq	-80(%rbp), %rdx
	xorl	%r13d, %r13d
	jmp	.LBB0_646
	.p2align	4, 0x90
.LBB0_653:
	incq	%r13
	addq	$4, %rdx
	addq	$12544, %r9
.LBB0_646:
	cmpq	$255, %r13
	jg	.LBB0_654
	movq	%r9, %r11
	movq	%rdx, %rbx
	xorl	%r12d, %r12d
	jmp	.LBB0_648
	.p2align	4, 0x90
.LBB0_652:
	incq	%r12
	addq	$4, %rbx
	addq	$224, %r11
.LBB0_648:
	testq	%r12, %r12
	jg	.LBB0_653
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_652
	.p2align	4, 0x90
.LBB0_651:
	movss	(%r11,%rcx,4), %xmm0
	imulq	$401408, %rax, %r15
	imulq	$3136, %rsi, %rdi
	addq	%r15, %rdi
	imulq	$56, %r8, %r15
	addq	%r10, %r15
	addq	%rdi, %r15
	mulss	(%rbx,%rcx,4), %xmm0
	addss	(%r14,%r15,4), %xmm0
	movss	%xmm0, (%r14,%r15,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_651
	jmp	.LBB0_652
.LBB0_658:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r14, %rcx
	jmp	.LBB0_659
	.p2align	4, 0x90
.LBB0_669:
	incq	%rax
	addq	$1605632, %rcx
.LBB0_659:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_670
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_661
	.p2align	4, 0x90
.LBB0_668:
	incq	%rsi
	addq	$12544, %rdx
.LBB0_661:
	cmpq	$127, %rsi
	jg	.LBB0_669
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_663
	.p2align	4, 0x90
.LBB0_667:
	incq	%r8
	addq	$224, %rdi
.LBB0_663:
	cmpq	$55, %r8
	jg	.LBB0_668
	xorl	%r9d, %r9d
	cmpq	$55, %r9
	jg	.LBB0_667
	.p2align	4, 0x90
.LBB0_666:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$55, %r9
	jle	.LBB0_666
	jmp	.LBB0_667
.LBB0_670:
	imulq	$1722368, -48(%rbp), %rdi
	orq	$64, %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -64(%rbp)
	jmp	.LBB0_671
	.p2align	4, 0x90
.LBB0_681:
	incq	%rcx
	addq	$1722368, %rdx
.LBB0_671:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_682
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_673
	.p2align	4, 0x90
.LBB0_680:
	incq	%rdi
	addq	$13456, %rsi
.LBB0_673:
	cmpq	$127, %rdi
	jg	.LBB0_681
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_675
	.p2align	4, 0x90
.LBB0_679:
	incq	%r9
	addq	$232, %r8
.LBB0_675:
	cmpq	$57, %r9
	jg	.LBB0_680
	xorl	%r10d, %r10d
	cmpq	$57, %r10
	jg	.LBB0_679
	.p2align	4, 0x90
.LBB0_678:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$57, %r10
	jle	.LBB0_678
	jmp	.LBB0_679
.LBB0_682:
	movq	%rsp, %r13
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$56, %edi
	movq	%rdi, -56(%rdx)
	movl	$128, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rbx
	movq	%rbx, -72(%rdx)
	movq	%r14, -88(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -96(%rdx)
	movq	%rdi, -48(%rdx)
	movl	$401408, %esi
	movq	%rsi, -40(%rdx)
	movl	$3136, %esi
	movq	%rsi, -32(%rdx)
	movq	%rdi, -24(%rdx)
	movl	$1, %esi
	movq	%rsi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rbx, -72(%rdx)
	movq	-64(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$58, -24(%rdx)
	movq	$3364, -32(%rdx)
	movq	$430592, -40(%rdx)
	movq	$56, -48(%rdx)
	movq	$56, -56(%rdx)
	movq	$128, -64(%rdx)
	movq	$59, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	%r13, %rsp
	imulq	$401408, %rbx, %rdi
	orq	$64, %rdi
	movq	%rdi, -136(%rbp)
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	xorl	%eax, %eax
	movq	%r15, %rcx
	movq	2776(%rbp), %r10
	jmp	.LBB0_683
	.p2align	4, 0x90
.LBB0_693:
	incq	%rax
	addq	$401408, %rcx
.LBB0_683:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_694
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_685
	.p2align	4, 0x90
.LBB0_692:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_685:
	cmpq	$127, %rsi
	jg	.LBB0_693
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_687
	.p2align	4, 0x90
.LBB0_691:
	incq	%r8
	addq	$112, %rdi
.LBB0_687:
	cmpq	$27, %r8
	jg	.LBB0_692
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_691
	.p2align	4, 0x90
.LBB0_690:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_690
	jmp	.LBB0_691
.LBB0_694:
	xorl	%eax, %eax
	jmp	.LBB0_695
	.p2align	4, 0x90
.LBB0_714:
	incq	%rax
	addq	$1722368, -64(%rbp)
.LBB0_695:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_715
	movq	2816(%rbp), %rcx
	movq	%rcx, -80(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_697
	.p2align	4, 0x90
.LBB0_713:
	incq	%rdx
	addq	$4608, -80(%rbp)
.LBB0_697:
	cmpq	$127, %rdx
	jg	.LBB0_714
	movq	-64(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_699
	.p2align	4, 0x90
.LBB0_712:
	incq	%rdi
	movq	-56(%rbp), %rcx
	addq	$464, %rcx
.LBB0_699:
	cmpq	$27, %rdi
	jg	.LBB0_713
	movq	%rcx, -56(%rbp)
	movq	%rcx, %r10
	xorl	%r9d, %r9d
	jmp	.LBB0_701
	.p2align	4, 0x90
.LBB0_711:
	incq	%r9
	movq	-88(%rbp), %r10
	addq	$8, %r10
.LBB0_701:
	cmpq	$27, %r9
	jg	.LBB0_712
	movq	%r10, -88(%rbp)
	movq	-80(%rbp), %r11
	xorl	%r14d, %r14d
	jmp	.LBB0_703
	.p2align	4, 0x90
.LBB0_710:
	incq	%r14
	addq	$36, %r11
	addq	$13456, %r10
.LBB0_703:
	cmpq	$127, %r14
	jg	.LBB0_711
	movq	%r10, %r12
	movq	%r11, %r13
	xorl	%ebx, %ebx
	jmp	.LBB0_705
	.p2align	4, 0x90
.LBB0_709:
	incq	%rbx
	addq	$12, %r13
	addq	$232, %r12
.LBB0_705:
	cmpq	$2, %rbx
	jg	.LBB0_710
	xorl	%esi, %esi
	cmpq	$2, %rsi
	jg	.LBB0_709
	.p2align	4, 0x90
.LBB0_708:
	movss	(%r12,%rsi,4), %xmm0
	imulq	$100352, %rax, %rcx
	imulq	$784, %rdx, %r8
	addq	%rcx, %r8
	leaq	(%rdi,%rdi,8), %rcx
	leaq	(%rcx,%rcx,2), %rcx
	addq	%rdi, %rcx
	addq	%r9, %r8
	addq	%rcx, %r8
	mulss	(%r13,%rsi,4), %xmm0
	addss	(%r15,%r8,4), %xmm0
	movss	%xmm0, (%r15,%r8,4)
	incq	%rsi
	cmpq	$2, %rsi
	jle	.LBB0_708
	jmp	.LBB0_709
.LBB0_715:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r15, %rcx
	jmp	.LBB0_716
	.p2align	4, 0x90
.LBB0_726:
	incq	%rax
	addq	$401408, %rcx
.LBB0_716:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_727
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_718
	.p2align	4, 0x90
.LBB0_725:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_718:
	cmpq	$127, %rsi
	jg	.LBB0_726
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_720
	.p2align	4, 0x90
.LBB0_724:
	incq	%r8
	addq	$112, %rdi
.LBB0_720:
	cmpq	$27, %r8
	jg	.LBB0_725
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_724
	.p2align	4, 0x90
.LBB0_723:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_723
	jmp	.LBB0_724
.LBB0_727:
	movq	-104(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r13
	addq	$63, %r13
	andq	$-64, %r13
	xorl	%eax, %eax
	movq	%r13, %rcx
	movq	2648(%rbp), %r10
	jmp	.LBB0_728
	.p2align	4, 0x90
.LBB0_738:
	incq	%rax
	addq	$1605632, %rcx
.LBB0_728:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_739
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_730
	.p2align	4, 0x90
.LBB0_737:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_730:
	cmpq	$511, %rsi
	jg	.LBB0_738
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_732
	.p2align	4, 0x90
.LBB0_736:
	incq	%r8
	addq	$112, %rdi
.LBB0_732:
	cmpq	$27, %r8
	jg	.LBB0_737
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_736
	.p2align	4, 0x90
.LBB0_735:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_735
	jmp	.LBB0_736
.LBB0_739:
	xorl	%eax, %eax
	jmp	.LBB0_740
	.p2align	4, 0x90
.LBB0_759:
	incq	%rax
	addq	$401408, %r15
.LBB0_740:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_760
	movq	2688(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_742
	.p2align	4, 0x90
.LBB0_758:
	incq	%rdx
	addq	$512, -64(%rbp)
.LBB0_742:
	cmpq	$511, %rdx
	jg	.LBB0_759
	movq	%r15, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_744
	.p2align	4, 0x90
.LBB0_757:
	incq	%rdi
	movq	-72(%rbp), %rcx
	addq	$112, %rcx
.LBB0_744:
	cmpq	$27, %rdi
	jg	.LBB0_758
	movq	%rcx, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_746
	.p2align	4, 0x90
.LBB0_756:
	incq	%r9
	movq	-56(%rbp), %rcx
	addq	$4, %rcx
.LBB0_746:
	cmpq	$27, %r9
	jg	.LBB0_757
	movq	%rcx, -56(%rbp)
	movq	%rcx, %r14
	movq	-64(%rbp), %r11
	xorl	%ebx, %ebx
	jmp	.LBB0_748
	.p2align	4, 0x90
.LBB0_755:
	incq	%rbx
	movq	-80(%rbp), %r11
	addq	$4, %r11
	movq	-88(%rbp), %r14
	addq	$3136, %r14
.LBB0_748:
	cmpq	$127, %rbx
	jg	.LBB0_756
	movq	%r14, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_750
	.p2align	4, 0x90
.LBB0_754:
	incq	%rsi
	addq	$4, %r11
	addq	$112, %r14
.LBB0_750:
	testq	%rsi, %rsi
	jg	.LBB0_755
	xorl	%r8d, %r8d
	testq	%r8, %r8
	jg	.LBB0_754
	.p2align	4, 0x90
.LBB0_753:
	movss	(%r14,%r8,4), %xmm0
	imulq	$401408, %rax, %r10
	imulq	$784, %rdx, %rcx
	addq	%r10, %rcx
	leaq	(%rdi,%rdi,8), %r10
	leaq	(%r10,%r10,2), %r10
	addq	%rdi, %r10
	addq	%r9, %rcx
	addq	%r10, %rcx
	mulss	(%r11,%r8,4), %xmm0
	addss	(%r13,%rcx,4), %xmm0
	movss	%xmm0, (%r13,%rcx,4)
	incq	%r8
	testq	%r8, %r8
	jle	.LBB0_753
	jmp	.LBB0_754
.LBB0_760:
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_761
	.p2align	4, 0x90
.LBB0_771:
	incq	%rax
	addq	$1605632, %rcx
.LBB0_761:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_772
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_763
	.p2align	4, 0x90
.LBB0_770:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_763:
	cmpq	$511, %rsi
	jg	.LBB0_771
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_765
	.p2align	4, 0x90
.LBB0_769:
	incq	%r8
	addq	$112, %rdi
.LBB0_765:
	cmpq	$27, %r8
	jg	.LBB0_770
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_769
	.p2align	4, 0x90
.LBB0_768:
	movq	3032(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_768
	jmp	.LBB0_769
.LBB0_772:
	xorl	%eax, %eax
	jmp	.LBB0_773
	.p2align	4, 0x90
.LBB0_792:
	incq	%rax
	addq	$3211264, -96(%rbp)
.LBB0_773:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_793
	movq	3112(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_775
	.p2align	4, 0x90
.LBB0_791:
	incq	%rdx
	addq	$1024, -56(%rbp)
.LBB0_775:
	cmpq	$511, %rdx
	jg	.LBB0_792
	movq	-96(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_777
	.p2align	4, 0x90
.LBB0_790:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$448, %rcx
.LBB0_777:
	cmpq	$27, %rdi
	jg	.LBB0_791
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_779
	.p2align	4, 0x90
.LBB0_789:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$8, %rcx
.LBB0_779:
	cmpq	$27, %r9
	jg	.LBB0_790
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r14
	movq	-56(%rbp), %r11
	xorl	%ebx, %ebx
	jmp	.LBB0_781
	.p2align	4, 0x90
.LBB0_788:
	incq	%rbx
	addq	$4, %r11
	movq	-88(%rbp), %r14
	addq	$12544, %r14
.LBB0_781:
	cmpq	$255, %rbx
	jg	.LBB0_789
	movq	%r14, -88(%rbp)
	movq	%r11, %r15
	xorl	%esi, %esi
	jmp	.LBB0_783
	.p2align	4, 0x90
.LBB0_787:
	incq	%rsi
	addq	$4, %r15
	addq	$224, %r14
.LBB0_783:
	testq	%rsi, %rsi
	jg	.LBB0_788
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_787
	.p2align	4, 0x90
.LBB0_786:
	movss	(%r14,%rcx,4), %xmm0
	imulq	$401408, %rax, %r8
	imulq	$784, %rdx, %r10
	addq	%r8, %r10
	leaq	(%rdi,%rdi,8), %r8
	leaq	(%r8,%r8,2), %r8
	addq	%rdi, %r8
	addq	%r9, %r10
	addq	%r8, %r10
	mulss	(%r15,%rcx,4), %xmm0
	addss	(%r12,%r10,4), %xmm0
	movss	%xmm0, (%r12,%r10,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_786
	jmp	.LBB0_787
.LBB0_793:
	xorl	%eax, %eax
	movq	%r13, %rcx
	jmp	.LBB0_794
	.p2align	4, 0x90
.LBB0_804:
	incq	%rax
	addq	$1605632, %r12
	addq	$1605632, %rcx
.LBB0_794:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_805
	movq	%rcx, %rdx
	movq	%r12, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_796
	.p2align	4, 0x90
.LBB0_803:
	incq	%rdi
	addq	$3136, %rsi
	addq	$3136, %rdx
.LBB0_796:
	cmpq	$511, %rdi
	jg	.LBB0_804
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_798
	.p2align	4, 0x90
.LBB0_802:
	incq	%r10
	addq	$112, %r9
	addq	$112, %r8
.LBB0_798:
	cmpq	$27, %r10
	jg	.LBB0_803
	xorl	%r11d, %r11d
	cmpq	$27, %r11
	jg	.LBB0_802
	.p2align	4, 0x90
.LBB0_801:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$27, %r11
	jle	.LBB0_801
	jmp	.LBB0_802
.LBB0_805:
	movq	-104(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_806
	.p2align	4, 0x90
.LBB0_816:
	incq	%rbx
	addq	$1605632, %rcx
	addq	$1605632, %r13
.LBB0_806:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_817
	movq	%r13, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_808
	.p2align	4, 0x90
.LBB0_815:
	incq	%rdi
	addq	$3136, %rsi
	addq	$3136, %rdx
.LBB0_808:
	cmpq	$511, %rdi
	jg	.LBB0_816
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_810
	.p2align	4, 0x90
.LBB0_814:
	incq	%r10
	addq	$112, %r9
	addq	$112, %r8
.LBB0_810:
	cmpq	$27, %r10
	jg	.LBB0_815
	xorl	%r11d, %r11d
	cmpq	$27, %r11
	jg	.LBB0_814
	.p2align	4, 0x90
.LBB0_813:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$27, %r11
	jle	.LBB0_813
	jmp	.LBB0_814
.LBB0_817:
	movq	%rax, -112(%rbp)
	movq	-136(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -120(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	2520(%rbp), %r10
	jmp	.LBB0_818
	.p2align	4, 0x90
.LBB0_828:
	incq	%rax
	addq	$401408, %rcx
.LBB0_818:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_829
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_820
	.p2align	4, 0x90
.LBB0_827:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_820:
	cmpq	$127, %rsi
	jg	.LBB0_828
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_822
	.p2align	4, 0x90
.LBB0_826:
	incq	%r8
	addq	$112, %rdi
.LBB0_822:
	cmpq	$27, %r8
	jg	.LBB0_827
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_826
	.p2align	4, 0x90
.LBB0_825:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_825
	jmp	.LBB0_826
.LBB0_829:
	xorl	%eax, %eax
	movq	-112(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	jmp	.LBB0_830
	.p2align	4, 0x90
.LBB0_849:
	incq	%rax
	addq	$1605632, -72(%rbp)
.LBB0_830:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_850
	movq	2560(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_832
	.p2align	4, 0x90
.LBB0_848:
	incq	%rsi
	addq	$2048, -56(%rbp)
.LBB0_832:
	cmpq	$127, %rsi
	jg	.LBB0_849
	movq	-72(%rbp), %r11
	xorl	%r8d, %r8d
	jmp	.LBB0_834
	.p2align	4, 0x90
.LBB0_847:
	incq	%r8
	movq	-64(%rbp), %r11
	addq	$112, %r11
.LBB0_834:
	cmpq	$27, %r8
	jg	.LBB0_848
	movq	%r11, -64(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_836
	.p2align	4, 0x90
.LBB0_846:
	incq	%r10
	movq	-80(%rbp), %r11
	addq	$4, %r11
.LBB0_836:
	cmpq	$27, %r10
	jg	.LBB0_847
	movq	%r11, -80(%rbp)
	movq	-56(%rbp), %rdx
	xorl	%r13d, %r13d
	jmp	.LBB0_838
	.p2align	4, 0x90
.LBB0_845:
	incq	%r13
	addq	$4, %rdx
	movq	-88(%rbp), %r11
	addq	$3136, %r11
.LBB0_838:
	cmpq	$511, %r13
	jg	.LBB0_846
	movq	%r11, -88(%rbp)
	movq	%rdx, %r14
	xorl	%ecx, %ecx
	jmp	.LBB0_840
	.p2align	4, 0x90
.LBB0_844:
	incq	%rcx
	addq	$4, %r14
	addq	$112, %r11
.LBB0_840:
	testq	%rcx, %rcx
	jg	.LBB0_845
	xorl	%r15d, %r15d
	testq	%r15, %r15
	jg	.LBB0_844
	.p2align	4, 0x90
.LBB0_843:
	movss	(%r11,%r15,4), %xmm0
	imulq	$100352, %rax, %rdi
	imulq	$784, %rsi, %r9
	addq	%rdi, %r9
	leaq	(%r8,%r8,8), %rdi
	leaq	(%rdi,%rdi,2), %rdi
	addq	%r8, %rdi
	addq	%r10, %r9
	addq	%rdi, %r9
	mulss	(%r14,%r15,4), %xmm0
	addss	(%rbx,%r9,4), %xmm0
	movss	%xmm0, (%rbx,%r9,4)
	incq	%r15
	testq	%r15, %r15
	jle	.LBB0_843
	jmp	.LBB0_844
.LBB0_850:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_851
	.p2align	4, 0x90
.LBB0_861:
	incq	%rax
	addq	$401408, %rcx
.LBB0_851:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_862
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_853
	.p2align	4, 0x90
.LBB0_860:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_853:
	cmpq	$127, %rsi
	jg	.LBB0_861
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_855
	.p2align	4, 0x90
.LBB0_859:
	incq	%r8
	addq	$112, %rdi
.LBB0_855:
	cmpq	$27, %r8
	jg	.LBB0_860
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_859
	.p2align	4, 0x90
.LBB0_858:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_858
	jmp	.LBB0_859
.LBB0_862:
	imulq	$460800, -48(%rbp), %rdi
	orq	$64, %rdi
	movq	%rdi, -96(%rbp)
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -72(%rbp)
	jmp	.LBB0_863
	.p2align	4, 0x90
.LBB0_873:
	incq	%rcx
	addq	$460800, %rdx
.LBB0_863:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_874
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_865
	.p2align	4, 0x90
.LBB0_872:
	incq	%rdi
	addq	$3600, %rsi
.LBB0_865:
	cmpq	$127, %rdi
	jg	.LBB0_873
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_867
	.p2align	4, 0x90
.LBB0_871:
	incq	%r9
	addq	$120, %r8
.LBB0_867:
	cmpq	$29, %r9
	jg	.LBB0_872
	xorl	%r10d, %r10d
	cmpq	$29, %r10
	jg	.LBB0_871
	.p2align	4, 0x90
.LBB0_870:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$29, %r10
	jle	.LBB0_870
	jmp	.LBB0_871
.LBB0_874:
	movq	%rsp, %r13
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$28, %r8d
	movq	%r8, -56(%rdx)
	movl	$128, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-120(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$100352, %edi
	movq	%rdi, -40(%rdx)
	movl	$784, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$30, -24(%rdx)
	movq	$900, -32(%rdx)
	movq	$115200, -40(%rdx)
	movq	$28, -48(%rdx)
	movq	$28, -56(%rdx)
	movq	$128, -64(%rdx)
	movq	$31, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r13, %rsp
	movq	%rbx, %rcx
	movq	2392(%rbp), %r10
	jmp	.LBB0_875
	.p2align	4, 0x90
.LBB0_885:
	incq	%rax
	addq	$401408, %rcx
.LBB0_875:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_886
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_877
	.p2align	4, 0x90
.LBB0_884:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_877:
	cmpq	$127, %rsi
	jg	.LBB0_885
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_879
	.p2align	4, 0x90
.LBB0_883:
	incq	%r8
	addq	$112, %rdi
.LBB0_879:
	cmpq	$27, %r8
	jg	.LBB0_884
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_883
	.p2align	4, 0x90
.LBB0_882:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_882
	jmp	.LBB0_883
.LBB0_886:
	xorl	%eax, %eax
	jmp	.LBB0_887
	.p2align	4, 0x90
.LBB0_906:
	incq	%rax
	addq	$460800, -72(%rbp)
.LBB0_887:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_907
	movq	2432(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_889
	.p2align	4, 0x90
.LBB0_905:
	incq	%rdx
	addq	$4608, -56(%rbp)
.LBB0_889:
	cmpq	$127, %rdx
	jg	.LBB0_906
	movq	-72(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_891
	.p2align	4, 0x90
.LBB0_904:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$120, %rcx
.LBB0_891:
	cmpq	$27, %rdi
	jg	.LBB0_905
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_893
	.p2align	4, 0x90
.LBB0_903:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_893:
	cmpq	$27, %r9
	jg	.LBB0_904
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r15
	movq	-56(%rbp), %rcx
	xorl	%r13d, %r13d
	jmp	.LBB0_895
	.p2align	4, 0x90
.LBB0_902:
	incq	%r13
	addq	$36, %rcx
	movq	-88(%rbp), %r15
	addq	$3600, %r15
.LBB0_895:
	cmpq	$127, %r13
	jg	.LBB0_903
	movq	%r15, -88(%rbp)
	movq	%rcx, %r11
	xorl	%esi, %esi
	jmp	.LBB0_897
	.p2align	4, 0x90
.LBB0_901:
	incq	%rsi
	addq	$12, %r11
	addq	$120, %r15
.LBB0_897:
	cmpq	$2, %rsi
	jg	.LBB0_902
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_901
	.p2align	4, 0x90
.LBB0_900:
	movss	(%r15,%r14,4), %xmm0
	imulq	$100352, %rax, %r8
	imulq	$784, %rdx, %r10
	addq	%r8, %r10
	leaq	(%rdi,%rdi,8), %r8
	leaq	(%r8,%r8,2), %r8
	addq	%rdi, %r8
	addq	%r9, %r10
	addq	%r8, %r10
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%r10,4), %xmm0
	movss	%xmm0, (%rbx,%r10,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_900
	jmp	.LBB0_901
.LBB0_907:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_908
	.p2align	4, 0x90
.LBB0_918:
	incq	%rax
	addq	$401408, %rcx
.LBB0_908:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_919
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_910
	.p2align	4, 0x90
.LBB0_917:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_910:
	cmpq	$127, %rsi
	jg	.LBB0_918
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_912
	.p2align	4, 0x90
.LBB0_916:
	incq	%r8
	addq	$112, %rdi
.LBB0_912:
	cmpq	$27, %r8
	jg	.LBB0_917
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_916
	.p2align	4, 0x90
.LBB0_915:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_915
	jmp	.LBB0_916
.LBB0_919:
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_920
	.p2align	4, 0x90
.LBB0_930:
	incq	%rax
	addq	$1605632, %rcx
.LBB0_920:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_931
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_922
	.p2align	4, 0x90
.LBB0_929:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_922:
	cmpq	$511, %rsi
	jg	.LBB0_930
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_924
	.p2align	4, 0x90
.LBB0_928:
	incq	%r8
	addq	$112, %rdi
.LBB0_924:
	cmpq	$27, %r8
	jg	.LBB0_929
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_928
	.p2align	4, 0x90
.LBB0_927:
	movq	2264(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_927
	jmp	.LBB0_928
.LBB0_931:
	xorl	%eax, %eax
	jmp	.LBB0_932
	.p2align	4, 0x90
.LBB0_951:
	incq	%rax
	addq	$401408, %rbx
.LBB0_932:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_952
	movq	2304(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_934
	.p2align	4, 0x90
.LBB0_950:
	incq	%rdx
	addq	$512, -56(%rbp)
.LBB0_934:
	cmpq	$511, %rdx
	jg	.LBB0_951
	movq	%rbx, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_936
	.p2align	4, 0x90
.LBB0_949:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$112, %rcx
.LBB0_936:
	cmpq	$27, %rdi
	jg	.LBB0_950
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_938
	.p2align	4, 0x90
.LBB0_948:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_938:
	cmpq	$27, %r9
	jg	.LBB0_949
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r15
	movq	-56(%rbp), %r11
	xorl	%r14d, %r14d
	jmp	.LBB0_940
	.p2align	4, 0x90
.LBB0_947:
	incq	%r14
	addq	$4, %r11
	movq	-88(%rbp), %r15
	addq	$3136, %r15
.LBB0_940:
	cmpq	$127, %r14
	jg	.LBB0_948
	movq	%r15, -88(%rbp)
	movq	%r11, %r13
	xorl	%esi, %esi
	jmp	.LBB0_942
	.p2align	4, 0x90
.LBB0_946:
	incq	%rsi
	addq	$4, %r13
	addq	$112, %r15
.LBB0_942:
	testq	%rsi, %rsi
	jg	.LBB0_947
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_946
	.p2align	4, 0x90
.LBB0_945:
	movss	(%r15,%rcx,4), %xmm0
	imulq	$401408, %rax, %r8
	imulq	$784, %rdx, %r10
	addq	%r8, %r10
	leaq	(%rdi,%rdi,8), %r8
	leaq	(%r8,%r8,2), %r8
	addq	%rdi, %r8
	addq	%r9, %r10
	addq	%r8, %r10
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r12,%r10,4), %xmm0
	movss	%xmm0, (%r12,%r10,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_945
	jmp	.LBB0_946
.LBB0_952:
	xorl	%eax, %eax
	movq	%r12, %rcx
	movq	-112(%rbp), %rbx
	jmp	.LBB0_953
	.p2align	4, 0x90
.LBB0_963:
	incq	%rax
	addq	$1605632, %rbx
	addq	$1605632, %rcx
.LBB0_953:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_964
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_955
	.p2align	4, 0x90
.LBB0_962:
	incq	%rdi
	addq	$3136, %rsi
	addq	$3136, %rdx
.LBB0_955:
	cmpq	$511, %rdi
	jg	.LBB0_963
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_957
	.p2align	4, 0x90
.LBB0_961:
	incq	%r10
	addq	$112, %r9
	addq	$112, %r8
.LBB0_957:
	cmpq	$27, %r10
	jg	.LBB0_962
	xorl	%r11d, %r11d
	cmpq	$27, %r11
	jg	.LBB0_961
	.p2align	4, 0x90
.LBB0_960:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$27, %r11
	jle	.LBB0_960
	jmp	.LBB0_961
.LBB0_964:
	movq	-104(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_965
	.p2align	4, 0x90
.LBB0_975:
	incq	%rbx
	addq	$1605632, %rcx
	addq	$1605632, %r12
.LBB0_965:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_976
	movq	%r12, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_967
	.p2align	4, 0x90
.LBB0_974:
	incq	%rdi
	addq	$3136, %rsi
	addq	$3136, %rdx
.LBB0_967:
	cmpq	$511, %rdi
	jg	.LBB0_975
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_969
	.p2align	4, 0x90
.LBB0_973:
	incq	%r10
	addq	$112, %r9
	addq	$112, %r8
.LBB0_969:
	cmpq	$27, %r10
	jg	.LBB0_974
	xorl	%r11d, %r11d
	cmpq	$27, %r11
	jg	.LBB0_973
	.p2align	4, 0x90
.LBB0_972:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$27, %r11
	jle	.LBB0_972
	jmp	.LBB0_973
.LBB0_976:
	movq	%rax, -112(%rbp)
	movq	-136(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -120(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	2136(%rbp), %r10
	jmp	.LBB0_977
	.p2align	4, 0x90
.LBB0_987:
	incq	%rax
	addq	$401408, %rcx
.LBB0_977:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_988
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_979
	.p2align	4, 0x90
.LBB0_986:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_979:
	cmpq	$127, %rsi
	jg	.LBB0_987
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_981
	.p2align	4, 0x90
.LBB0_985:
	incq	%r8
	addq	$112, %rdi
.LBB0_981:
	cmpq	$27, %r8
	jg	.LBB0_986
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_985
	.p2align	4, 0x90
.LBB0_984:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_984
	jmp	.LBB0_985
.LBB0_988:
	xorl	%eax, %eax
	movq	-112(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	jmp	.LBB0_989
	.p2align	4, 0x90
.LBB0_1008:
	incq	%rax
	addq	$1605632, -72(%rbp)
.LBB0_989:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1009
	movq	2176(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_991
	.p2align	4, 0x90
.LBB0_1007:
	incq	%rsi
	addq	$2048, -56(%rbp)
.LBB0_991:
	cmpq	$127, %rsi
	jg	.LBB0_1008
	movq	-72(%rbp), %r11
	xorl	%r8d, %r8d
	jmp	.LBB0_993
	.p2align	4, 0x90
.LBB0_1006:
	incq	%r8
	movq	-64(%rbp), %r11
	addq	$112, %r11
.LBB0_993:
	cmpq	$27, %r8
	jg	.LBB0_1007
	movq	%r11, -64(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_995
	.p2align	4, 0x90
.LBB0_1005:
	incq	%r10
	movq	-80(%rbp), %r11
	addq	$4, %r11
.LBB0_995:
	cmpq	$27, %r10
	jg	.LBB0_1006
	movq	%r11, -80(%rbp)
	movq	-56(%rbp), %rdx
	xorl	%r12d, %r12d
	jmp	.LBB0_997
	.p2align	4, 0x90
.LBB0_1004:
	incq	%r12
	addq	$4, %rdx
	movq	-88(%rbp), %r11
	addq	$3136, %r11
.LBB0_997:
	cmpq	$511, %r12
	jg	.LBB0_1005
	movq	%r11, -88(%rbp)
	movq	%rdx, %r14
	xorl	%ecx, %ecx
	jmp	.LBB0_999
	.p2align	4, 0x90
.LBB0_1003:
	incq	%rcx
	addq	$4, %r14
	addq	$112, %r11
.LBB0_999:
	testq	%rcx, %rcx
	jg	.LBB0_1004
	xorl	%r13d, %r13d
	testq	%r13, %r13
	jg	.LBB0_1003
	.p2align	4, 0x90
.LBB0_1002:
	movss	(%r11,%r13,4), %xmm0
	imulq	$100352, %rax, %rdi
	imulq	$784, %rsi, %r9
	addq	%rdi, %r9
	leaq	(%r8,%r8,8), %rdi
	leaq	(%rdi,%rdi,2), %rdi
	addq	%r8, %rdi
	addq	%r10, %r9
	addq	%rdi, %r9
	mulss	(%r14,%r13,4), %xmm0
	addss	(%rbx,%r9,4), %xmm0
	movss	%xmm0, (%rbx,%r9,4)
	incq	%r13
	testq	%r13, %r13
	jle	.LBB0_1002
	jmp	.LBB0_1003
.LBB0_1009:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1010
	.p2align	4, 0x90
.LBB0_1020:
	incq	%rax
	addq	$401408, %rcx
.LBB0_1010:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1021
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1012
	.p2align	4, 0x90
.LBB0_1019:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1012:
	cmpq	$127, %rsi
	jg	.LBB0_1020
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1014
	.p2align	4, 0x90
.LBB0_1018:
	incq	%r8
	addq	$112, %rdi
.LBB0_1014:
	cmpq	$27, %r8
	jg	.LBB0_1019
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1018
	.p2align	4, 0x90
.LBB0_1017:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1017
	jmp	.LBB0_1018
.LBB0_1021:
	movq	-96(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -72(%rbp)
	jmp	.LBB0_1022
	.p2align	4, 0x90
.LBB0_1032:
	incq	%rcx
	addq	$460800, %rdx
.LBB0_1022:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_1033
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1024
	.p2align	4, 0x90
.LBB0_1031:
	incq	%rdi
	addq	$3600, %rsi
.LBB0_1024:
	cmpq	$127, %rdi
	jg	.LBB0_1032
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_1026
	.p2align	4, 0x90
.LBB0_1030:
	incq	%r9
	addq	$120, %r8
.LBB0_1026:
	cmpq	$29, %r9
	jg	.LBB0_1031
	xorl	%r10d, %r10d
	cmpq	$29, %r10
	jg	.LBB0_1030
	.p2align	4, 0x90
.LBB0_1029:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$29, %r10
	jle	.LBB0_1029
	jmp	.LBB0_1030
.LBB0_1033:
	movq	%rsp, %r12
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$28, %r8d
	movq	%r8, -56(%rdx)
	movl	$128, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-120(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$100352, %edi
	movq	%rdi, -40(%rdx)
	movl	$784, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$30, -24(%rdx)
	movq	$900, -32(%rdx)
	movq	$115200, -40(%rdx)
	movq	$28, -48(%rdx)
	movq	$28, -56(%rdx)
	movq	$128, -64(%rdx)
	movq	$31, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r12, %rsp
	movq	%rbx, %rcx
	movq	2008(%rbp), %r10
	jmp	.LBB0_1034
	.p2align	4, 0x90
.LBB0_1044:
	incq	%rax
	addq	$401408, %rcx
.LBB0_1034:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1045
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1036
	.p2align	4, 0x90
.LBB0_1043:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1036:
	cmpq	$127, %rsi
	jg	.LBB0_1044
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1038
	.p2align	4, 0x90
.LBB0_1042:
	incq	%r8
	addq	$112, %rdi
.LBB0_1038:
	cmpq	$27, %r8
	jg	.LBB0_1043
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1042
	.p2align	4, 0x90
.LBB0_1041:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1041
	jmp	.LBB0_1042
.LBB0_1045:
	xorl	%eax, %eax
	jmp	.LBB0_1046
	.p2align	4, 0x90
.LBB0_1065:
	incq	%rax
	addq	$460800, -72(%rbp)
.LBB0_1046:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1066
	movq	2048(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1048
	.p2align	4, 0x90
.LBB0_1064:
	incq	%rdx
	addq	$4608, -56(%rbp)
.LBB0_1048:
	cmpq	$127, %rdx
	jg	.LBB0_1065
	movq	-72(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_1050
	.p2align	4, 0x90
.LBB0_1063:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$120, %rcx
.LBB0_1050:
	cmpq	$27, %rdi
	jg	.LBB0_1064
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1052
	.p2align	4, 0x90
.LBB0_1062:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_1052:
	cmpq	$27, %r9
	jg	.LBB0_1063
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r13
	movq	-56(%rbp), %rcx
	xorl	%r12d, %r12d
	jmp	.LBB0_1054
	.p2align	4, 0x90
.LBB0_1061:
	incq	%r12
	addq	$36, %rcx
	movq	-88(%rbp), %r13
	addq	$3600, %r13
.LBB0_1054:
	cmpq	$127, %r12
	jg	.LBB0_1062
	movq	%r13, -88(%rbp)
	movq	%rcx, %r11
	xorl	%esi, %esi
	jmp	.LBB0_1056
	.p2align	4, 0x90
.LBB0_1060:
	incq	%rsi
	addq	$12, %r11
	addq	$120, %r13
.LBB0_1056:
	cmpq	$2, %rsi
	jg	.LBB0_1061
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_1060
	.p2align	4, 0x90
.LBB0_1059:
	movss	(%r13,%r14,4), %xmm0
	imulq	$100352, %rax, %r8
	imulq	$784, %rdx, %r10
	addq	%r8, %r10
	leaq	(%rdi,%rdi,8), %r8
	leaq	(%r8,%r8,2), %r8
	addq	%rdi, %r8
	addq	%r9, %r10
	addq	%r8, %r10
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%r10,4), %xmm0
	movss	%xmm0, (%rbx,%r10,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_1059
	jmp	.LBB0_1060
.LBB0_1066:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1067
	.p2align	4, 0x90
.LBB0_1077:
	incq	%rax
	addq	$401408, %rcx
.LBB0_1067:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1078
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1069
	.p2align	4, 0x90
.LBB0_1076:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1069:
	cmpq	$127, %rsi
	jg	.LBB0_1077
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1071
	.p2align	4, 0x90
.LBB0_1075:
	incq	%r8
	addq	$112, %rdi
.LBB0_1071:
	cmpq	$27, %r8
	jg	.LBB0_1076
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1075
	.p2align	4, 0x90
.LBB0_1074:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1074
	jmp	.LBB0_1075
.LBB0_1078:
	xorl	%eax, %eax
	movq	%r15, %rcx
	jmp	.LBB0_1079
	.p2align	4, 0x90
.LBB0_1089:
	incq	%rax
	addq	$1605632, %rcx
.LBB0_1079:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1090
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1081
	.p2align	4, 0x90
.LBB0_1088:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1081:
	cmpq	$511, %rsi
	jg	.LBB0_1089
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1083
	.p2align	4, 0x90
.LBB0_1087:
	incq	%r8
	addq	$112, %rdi
.LBB0_1083:
	cmpq	$27, %r8
	jg	.LBB0_1088
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1087
	.p2align	4, 0x90
.LBB0_1086:
	movq	1880(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1086
	jmp	.LBB0_1087
.LBB0_1090:
	xorl	%eax, %eax
	jmp	.LBB0_1091
	.p2align	4, 0x90
.LBB0_1110:
	incq	%rax
	addq	$401408, %rbx
.LBB0_1091:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1111
	movq	1920(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1093
	.p2align	4, 0x90
.LBB0_1109:
	incq	%rdx
	addq	$512, -56(%rbp)
.LBB0_1093:
	cmpq	$511, %rdx
	jg	.LBB0_1110
	movq	%rbx, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_1095
	.p2align	4, 0x90
.LBB0_1108:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$112, %rcx
.LBB0_1095:
	cmpq	$27, %rdi
	jg	.LBB0_1109
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1097
	.p2align	4, 0x90
.LBB0_1107:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_1097:
	cmpq	$27, %r9
	jg	.LBB0_1108
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r12
	movq	-56(%rbp), %r11
	xorl	%r14d, %r14d
	jmp	.LBB0_1099
	.p2align	4, 0x90
.LBB0_1106:
	incq	%r14
	addq	$4, %r11
	movq	-88(%rbp), %r12
	addq	$3136, %r12
.LBB0_1099:
	cmpq	$127, %r14
	jg	.LBB0_1107
	movq	%r12, -88(%rbp)
	movq	%r11, %r13
	xorl	%esi, %esi
	jmp	.LBB0_1101
	.p2align	4, 0x90
.LBB0_1105:
	incq	%rsi
	addq	$4, %r13
	addq	$112, %r12
.LBB0_1101:
	testq	%rsi, %rsi
	jg	.LBB0_1106
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_1105
	.p2align	4, 0x90
.LBB0_1104:
	movss	(%r12,%rcx,4), %xmm0
	imulq	$401408, %rax, %r8
	imulq	$784, %rdx, %r10
	addq	%r8, %r10
	leaq	(%rdi,%rdi,8), %r8
	leaq	(%r8,%r8,2), %r8
	addq	%rdi, %r8
	addq	%r9, %r10
	addq	%r8, %r10
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r15,%r10,4), %xmm0
	movss	%xmm0, (%r15,%r10,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_1104
	jmp	.LBB0_1105
.LBB0_1111:
	xorl	%eax, %eax
	movq	%r15, %rcx
	movq	-112(%rbp), %rbx
	jmp	.LBB0_1112
	.p2align	4, 0x90
.LBB0_1122:
	incq	%rax
	addq	$1605632, %rbx
	addq	$1605632, %rcx
.LBB0_1112:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1123
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1114
	.p2align	4, 0x90
.LBB0_1121:
	incq	%rdi
	addq	$3136, %rsi
	addq	$3136, %rdx
.LBB0_1114:
	cmpq	$511, %rdi
	jg	.LBB0_1122
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1116
	.p2align	4, 0x90
.LBB0_1120:
	incq	%r10
	addq	$112, %r9
	addq	$112, %r8
.LBB0_1116:
	cmpq	$27, %r10
	jg	.LBB0_1121
	xorl	%r11d, %r11d
	cmpq	$27, %r11
	jg	.LBB0_1120
	.p2align	4, 0x90
.LBB0_1119:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$27, %r11
	jle	.LBB0_1119
	jmp	.LBB0_1120
.LBB0_1123:
	movq	-104(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_1124
	.p2align	4, 0x90
.LBB0_1134:
	incq	%rbx
	addq	$1605632, %rcx
	addq	$1605632, %r15
.LBB0_1124:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_1135
	movq	%r15, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1126
	.p2align	4, 0x90
.LBB0_1133:
	incq	%rdi
	addq	$3136, %rsi
	addq	$3136, %rdx
.LBB0_1126:
	cmpq	$511, %rdi
	jg	.LBB0_1134
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1128
	.p2align	4, 0x90
.LBB0_1132:
	incq	%r10
	addq	$112, %r9
	addq	$112, %r8
.LBB0_1128:
	cmpq	$27, %r10
	jg	.LBB0_1133
	xorl	%r11d, %r11d
	cmpq	$27, %r11
	jg	.LBB0_1132
	.p2align	4, 0x90
.LBB0_1131:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$27, %r11
	jle	.LBB0_1131
	jmp	.LBB0_1132
.LBB0_1135:
	movq	%rax, -112(%rbp)
	movq	-136(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -120(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	1752(%rbp), %r10
	jmp	.LBB0_1136
	.p2align	4, 0x90
.LBB0_1146:
	incq	%rax
	addq	$401408, %rcx
.LBB0_1136:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1147
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1138
	.p2align	4, 0x90
.LBB0_1145:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1138:
	cmpq	$127, %rsi
	jg	.LBB0_1146
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1140
	.p2align	4, 0x90
.LBB0_1144:
	incq	%r8
	addq	$112, %rdi
.LBB0_1140:
	cmpq	$27, %r8
	jg	.LBB0_1145
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1144
	.p2align	4, 0x90
.LBB0_1143:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1143
	jmp	.LBB0_1144
.LBB0_1147:
	xorl	%eax, %eax
	movq	-112(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	jmp	.LBB0_1148
	.p2align	4, 0x90
.LBB0_1167:
	incq	%rax
	addq	$1605632, -72(%rbp)
.LBB0_1148:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1168
	movq	1792(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1150
	.p2align	4, 0x90
.LBB0_1166:
	incq	%rsi
	addq	$2048, -56(%rbp)
.LBB0_1150:
	cmpq	$127, %rsi
	jg	.LBB0_1167
	movq	-72(%rbp), %r11
	xorl	%r8d, %r8d
	jmp	.LBB0_1152
	.p2align	4, 0x90
.LBB0_1165:
	incq	%r8
	movq	-64(%rbp), %r11
	addq	$112, %r11
.LBB0_1152:
	cmpq	$27, %r8
	jg	.LBB0_1166
	movq	%r11, -64(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_1154
	.p2align	4, 0x90
.LBB0_1164:
	incq	%r10
	movq	-80(%rbp), %r11
	addq	$4, %r11
.LBB0_1154:
	cmpq	$27, %r10
	jg	.LBB0_1165
	movq	%r11, -80(%rbp)
	movq	-56(%rbp), %rdx
	xorl	%r15d, %r15d
	jmp	.LBB0_1156
	.p2align	4, 0x90
.LBB0_1163:
	incq	%r15
	addq	$4, %rdx
	movq	-88(%rbp), %r11
	addq	$3136, %r11
.LBB0_1156:
	cmpq	$511, %r15
	jg	.LBB0_1164
	movq	%r11, -88(%rbp)
	movq	%rdx, %r14
	xorl	%ecx, %ecx
	jmp	.LBB0_1158
	.p2align	4, 0x90
.LBB0_1162:
	incq	%rcx
	addq	$4, %r14
	addq	$112, %r11
.LBB0_1158:
	testq	%rcx, %rcx
	jg	.LBB0_1163
	xorl	%r13d, %r13d
	testq	%r13, %r13
	jg	.LBB0_1162
	.p2align	4, 0x90
.LBB0_1161:
	movss	(%r11,%r13,4), %xmm0
	imulq	$100352, %rax, %rdi
	imulq	$784, %rsi, %r9
	addq	%rdi, %r9
	leaq	(%r8,%r8,8), %rdi
	leaq	(%rdi,%rdi,2), %rdi
	addq	%r8, %rdi
	addq	%r10, %r9
	addq	%rdi, %r9
	mulss	(%r14,%r13,4), %xmm0
	addss	(%rbx,%r9,4), %xmm0
	movss	%xmm0, (%rbx,%r9,4)
	incq	%r13
	testq	%r13, %r13
	jle	.LBB0_1161
	jmp	.LBB0_1162
.LBB0_1168:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1169
	.p2align	4, 0x90
.LBB0_1179:
	incq	%rax
	addq	$401408, %rcx
.LBB0_1169:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1180
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1171
	.p2align	4, 0x90
.LBB0_1178:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1171:
	cmpq	$127, %rsi
	jg	.LBB0_1179
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1173
	.p2align	4, 0x90
.LBB0_1177:
	incq	%r8
	addq	$112, %rdi
.LBB0_1173:
	cmpq	$27, %r8
	jg	.LBB0_1178
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1177
	.p2align	4, 0x90
.LBB0_1176:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1176
	jmp	.LBB0_1177
.LBB0_1180:
	movq	-96(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -72(%rbp)
	jmp	.LBB0_1181
	.p2align	4, 0x90
.LBB0_1191:
	incq	%rcx
	addq	$460800, %rdx
.LBB0_1181:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_1192
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1183
	.p2align	4, 0x90
.LBB0_1190:
	incq	%rdi
	addq	$3600, %rsi
.LBB0_1183:
	cmpq	$127, %rdi
	jg	.LBB0_1191
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_1185
	.p2align	4, 0x90
.LBB0_1189:
	incq	%r9
	addq	$120, %r8
.LBB0_1185:
	cmpq	$29, %r9
	jg	.LBB0_1190
	xorl	%r10d, %r10d
	cmpq	$29, %r10
	jg	.LBB0_1189
	.p2align	4, 0x90
.LBB0_1188:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$29, %r10
	jle	.LBB0_1188
	jmp	.LBB0_1189
.LBB0_1192:
	movq	%rsp, %r15
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$28, %r8d
	movq	%r8, -56(%rdx)
	movl	$128, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-120(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$100352, %edi
	movq	%rdi, -40(%rdx)
	movl	$784, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$30, -24(%rdx)
	movq	$900, -32(%rdx)
	movq	$115200, -40(%rdx)
	movq	$28, -48(%rdx)
	movq	$28, -56(%rdx)
	movq	$128, -64(%rdx)
	movq	$31, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r15, %rsp
	movq	%rbx, %rcx
	movq	4480(%rbp), %r10
	jmp	.LBB0_1193
	.p2align	4, 0x90
.LBB0_1203:
	incq	%rax
	addq	$401408, %rcx
.LBB0_1193:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1204
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1195
	.p2align	4, 0x90
.LBB0_1202:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1195:
	cmpq	$127, %rsi
	jg	.LBB0_1203
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1197
	.p2align	4, 0x90
.LBB0_1201:
	incq	%r8
	addq	$112, %rdi
.LBB0_1197:
	cmpq	$27, %r8
	jg	.LBB0_1202
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1201
	.p2align	4, 0x90
.LBB0_1200:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1200
	jmp	.LBB0_1201
.LBB0_1204:
	xorl	%eax, %eax
	jmp	.LBB0_1205
	.p2align	4, 0x90
.LBB0_1224:
	incq	%rax
	addq	$460800, -72(%rbp)
.LBB0_1205:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1225
	movq	4520(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1207
	.p2align	4, 0x90
.LBB0_1223:
	incq	%rdx
	addq	$4608, -56(%rbp)
.LBB0_1207:
	cmpq	$127, %rdx
	jg	.LBB0_1224
	movq	-72(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_1209
	.p2align	4, 0x90
.LBB0_1222:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$120, %rcx
.LBB0_1209:
	cmpq	$27, %rdi
	jg	.LBB0_1223
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1211
	.p2align	4, 0x90
.LBB0_1221:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_1211:
	cmpq	$27, %r9
	jg	.LBB0_1222
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r13
	movq	-56(%rbp), %rcx
	xorl	%r15d, %r15d
	jmp	.LBB0_1213
	.p2align	4, 0x90
.LBB0_1220:
	incq	%r15
	addq	$36, %rcx
	movq	-88(%rbp), %r13
	addq	$3600, %r13
.LBB0_1213:
	cmpq	$127, %r15
	jg	.LBB0_1221
	movq	%r13, -88(%rbp)
	movq	%rcx, %r11
	xorl	%esi, %esi
	jmp	.LBB0_1215
	.p2align	4, 0x90
.LBB0_1219:
	incq	%rsi
	addq	$12, %r11
	addq	$120, %r13
.LBB0_1215:
	cmpq	$2, %rsi
	jg	.LBB0_1220
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_1219
	.p2align	4, 0x90
.LBB0_1218:
	movss	(%r13,%r14,4), %xmm0
	imulq	$100352, %rax, %r8
	imulq	$784, %rdx, %r10
	addq	%r8, %r10
	leaq	(%rdi,%rdi,8), %r8
	leaq	(%r8,%r8,2), %r8
	addq	%rdi, %r8
	addq	%r9, %r10
	addq	%r8, %r10
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%r10,4), %xmm0
	movss	%xmm0, (%rbx,%r10,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_1218
	jmp	.LBB0_1219
.LBB0_1225:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1226
	.p2align	4, 0x90
.LBB0_1236:
	incq	%rax
	addq	$401408, %rcx
.LBB0_1226:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1237
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1228
	.p2align	4, 0x90
.LBB0_1235:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1228:
	cmpq	$127, %rsi
	jg	.LBB0_1236
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1230
	.p2align	4, 0x90
.LBB0_1234:
	incq	%r8
	addq	$112, %rdi
.LBB0_1230:
	cmpq	$27, %r8
	jg	.LBB0_1235
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1234
	.p2align	4, 0x90
.LBB0_1233:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1233
	jmp	.LBB0_1234
.LBB0_1237:
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_1238
	.p2align	4, 0x90
.LBB0_1248:
	incq	%rax
	addq	$1605632, %rcx
.LBB0_1238:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1249
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1240
	.p2align	4, 0x90
.LBB0_1247:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1240:
	cmpq	$511, %rsi
	jg	.LBB0_1248
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1242
	.p2align	4, 0x90
.LBB0_1246:
	incq	%r8
	addq	$112, %rdi
.LBB0_1242:
	cmpq	$27, %r8
	jg	.LBB0_1247
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1246
	.p2align	4, 0x90
.LBB0_1245:
	movq	4352(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1245
	jmp	.LBB0_1246
.LBB0_1249:
	xorl	%eax, %eax
	jmp	.LBB0_1250
	.p2align	4, 0x90
.LBB0_1269:
	incq	%rax
	addq	$401408, %rbx
.LBB0_1250:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1270
	movq	4392(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1252
	.p2align	4, 0x90
.LBB0_1268:
	incq	%rdx
	addq	$512, -56(%rbp)
.LBB0_1252:
	cmpq	$511, %rdx
	jg	.LBB0_1269
	movq	%rbx, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_1254
	.p2align	4, 0x90
.LBB0_1267:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$112, %rcx
.LBB0_1254:
	cmpq	$27, %rdi
	jg	.LBB0_1268
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1256
	.p2align	4, 0x90
.LBB0_1266:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_1256:
	cmpq	$27, %r9
	jg	.LBB0_1267
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r15
	movq	-56(%rbp), %r11
	xorl	%r14d, %r14d
	jmp	.LBB0_1258
	.p2align	4, 0x90
.LBB0_1265:
	incq	%r14
	addq	$4, %r11
	movq	-88(%rbp), %r15
	addq	$3136, %r15
.LBB0_1258:
	cmpq	$127, %r14
	jg	.LBB0_1266
	movq	%r15, -88(%rbp)
	movq	%r11, %r13
	xorl	%esi, %esi
	jmp	.LBB0_1260
	.p2align	4, 0x90
.LBB0_1264:
	incq	%rsi
	addq	$4, %r13
	addq	$112, %r15
.LBB0_1260:
	testq	%rsi, %rsi
	jg	.LBB0_1265
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_1264
	.p2align	4, 0x90
.LBB0_1263:
	movss	(%r15,%rcx,4), %xmm0
	imulq	$401408, %rax, %r8
	imulq	$784, %rdx, %r10
	addq	%r8, %r10
	leaq	(%rdi,%rdi,8), %r8
	leaq	(%r8,%r8,2), %r8
	addq	%rdi, %r8
	addq	%r9, %r10
	addq	%r8, %r10
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r12,%r10,4), %xmm0
	movss	%xmm0, (%r12,%r10,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_1263
	jmp	.LBB0_1264
.LBB0_1270:
	xorl	%eax, %eax
	movq	%r12, %rcx
	movq	-112(%rbp), %rbx
	jmp	.LBB0_1271
	.p2align	4, 0x90
.LBB0_1281:
	incq	%rax
	addq	$1605632, %rbx
	addq	$1605632, %rcx
.LBB0_1271:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1282
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1273
	.p2align	4, 0x90
.LBB0_1280:
	incq	%rdi
	addq	$3136, %rsi
	addq	$3136, %rdx
.LBB0_1273:
	cmpq	$511, %rdi
	jg	.LBB0_1281
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1275
	.p2align	4, 0x90
.LBB0_1279:
	incq	%r10
	addq	$112, %r9
	addq	$112, %r8
.LBB0_1275:
	cmpq	$27, %r10
	jg	.LBB0_1280
	xorl	%r11d, %r11d
	cmpq	$27, %r11
	jg	.LBB0_1279
	.p2align	4, 0x90
.LBB0_1278:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$27, %r11
	jle	.LBB0_1278
	jmp	.LBB0_1279
.LBB0_1282:
	movq	-104(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rcx
	addq	$63, %rcx
	andq	$-64, %rcx
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rcx, -104(%rbp)
	jmp	.LBB0_1283
	.p2align	4, 0x90
.LBB0_1293:
	incq	%rax
	addq	$1605632, %rcx
	addq	$1605632, %r12
.LBB0_1283:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1294
	movq	%r12, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1285
	.p2align	4, 0x90
.LBB0_1292:
	incq	%rdi
	addq	$3136, %rsi
	addq	$3136, %rdx
.LBB0_1285:
	cmpq	$511, %rdi
	jg	.LBB0_1293
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1287
	.p2align	4, 0x90
.LBB0_1291:
	incq	%r10
	addq	$112, %r9
	addq	$112, %r8
.LBB0_1287:
	cmpq	$27, %r10
	jg	.LBB0_1292
	xorl	%r11d, %r11d
	cmpq	$27, %r11
	jg	.LBB0_1291
	.p2align	4, 0x90
.LBB0_1290:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$27, %r11
	jle	.LBB0_1290
	jmp	.LBB0_1291
.LBB0_1294:
	movq	-128(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -96(%rbp)
	leaq	63(%rax), %r14
	andq	$-64, %r14
	xorl	%eax, %eax
	movq	%r14, %rcx
	movq	4096(%rbp), %r10
	jmp	.LBB0_1295
	.p2align	4, 0x90
.LBB0_1305:
	incq	%rax
	addq	$802816, %rcx
.LBB0_1295:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1306
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1297
	.p2align	4, 0x90
.LBB0_1304:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1297:
	cmpq	$255, %rsi
	jg	.LBB0_1305
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1299
	.p2align	4, 0x90
.LBB0_1303:
	incq	%r8
	addq	$112, %rdi
.LBB0_1299:
	cmpq	$27, %r8
	jg	.LBB0_1304
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1303
	.p2align	4, 0x90
.LBB0_1302:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1302
	jmp	.LBB0_1303
.LBB0_1306:
	xorl	%eax, %eax
	movq	-104(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	jmp	.LBB0_1307
	.p2align	4, 0x90
.LBB0_1326:
	incq	%rax
	addq	$1605632, -64(%rbp)
.LBB0_1307:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1327
	movq	4136(%rbp), %rcx
	movq	%rcx, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1309
	.p2align	4, 0x90
.LBB0_1325:
	incq	%rsi
	addq	$2048, -80(%rbp)
.LBB0_1309:
	cmpq	$255, %rsi
	jg	.LBB0_1326
	movq	-64(%rbp), %r9
	xorl	%r8d, %r8d
	jmp	.LBB0_1311
	.p2align	4, 0x90
.LBB0_1324:
	incq	%r8
	movq	-56(%rbp), %r9
	addq	$112, %r9
.LBB0_1311:
	cmpq	$27, %r8
	jg	.LBB0_1325
	movq	%r9, -56(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_1313
	.p2align	4, 0x90
.LBB0_1323:
	incq	%r10
	movq	-88(%rbp), %r9
	addq	$4, %r9
.LBB0_1313:
	cmpq	$27, %r10
	jg	.LBB0_1324
	movq	%r9, -88(%rbp)
	movq	-80(%rbp), %rdx
	xorl	%r13d, %r13d
	jmp	.LBB0_1315
	.p2align	4, 0x90
.LBB0_1322:
	incq	%r13
	addq	$4, %rdx
	addq	$3136, %r9
.LBB0_1315:
	cmpq	$511, %r13
	jg	.LBB0_1323
	movq	%r9, %r11
	movq	%rdx, %rbx
	xorl	%r12d, %r12d
	jmp	.LBB0_1317
	.p2align	4, 0x90
.LBB0_1321:
	incq	%r12
	addq	$4, %rbx
	addq	$112, %r11
.LBB0_1317:
	testq	%r12, %r12
	jg	.LBB0_1322
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_1321
	.p2align	4, 0x90
.LBB0_1320:
	movss	(%r11,%rcx,4), %xmm0
	imulq	$200704, %rax, %r15
	imulq	$784, %rsi, %rdi
	addq	%r15, %rdi
	leaq	(%r8,%r8,8), %r15
	leaq	(%r15,%r15,2), %r15
	addq	%r8, %r15
	addq	%r10, %rdi
	addq	%r15, %rdi
	mulss	(%rbx,%rcx,4), %xmm0
	addss	(%r14,%rdi,4), %xmm0
	movss	%xmm0, (%r14,%rdi,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_1320
	jmp	.LBB0_1321
.LBB0_1327:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r14, %rcx
	jmp	.LBB0_1328
	.p2align	4, 0x90
.LBB0_1338:
	incq	%rax
	addq	$802816, %rcx
.LBB0_1328:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1339
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1330
	.p2align	4, 0x90
.LBB0_1337:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_1330:
	cmpq	$255, %rsi
	jg	.LBB0_1338
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1332
	.p2align	4, 0x90
.LBB0_1336:
	incq	%r8
	addq	$112, %rdi
.LBB0_1332:
	cmpq	$27, %r8
	jg	.LBB0_1337
	xorl	%r9d, %r9d
	cmpq	$27, %r9
	jg	.LBB0_1336
	.p2align	4, 0x90
.LBB0_1335:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$27, %r9
	jle	.LBB0_1335
	jmp	.LBB0_1336
.LBB0_1339:
	imulq	$921600, -48(%rbp), %rdi
	orq	$64, %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -72(%rbp)
	jmp	.LBB0_1340
	.p2align	4, 0x90
.LBB0_1350:
	incq	%rcx
	addq	$921600, %rdx
.LBB0_1340:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_1351
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1342
	.p2align	4, 0x90
.LBB0_1349:
	incq	%rdi
	addq	$3600, %rsi
.LBB0_1342:
	cmpq	$255, %rdi
	jg	.LBB0_1350
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_1344
	.p2align	4, 0x90
.LBB0_1348:
	incq	%r9
	addq	$120, %r8
.LBB0_1344:
	cmpq	$29, %r9
	jg	.LBB0_1349
	xorl	%r10d, %r10d
	cmpq	$29, %r10
	jg	.LBB0_1348
	.p2align	4, 0x90
.LBB0_1347:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$29, %r10
	jle	.LBB0_1347
	jmp	.LBB0_1348
.LBB0_1351:
	movq	%rsp, %r13
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$28, %edi
	movq	%rdi, -56(%rdx)
	movl	$256, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rbx
	movq	%rbx, -72(%rdx)
	movq	%r14, -88(%rdx)
	movq	-96(%rbp), %rsi
	movq	%rsi, -96(%rdx)
	movq	%rdi, -48(%rdx)
	movl	$200704, %esi
	movq	%rsi, -40(%rdx)
	movl	$784, %esi
	movq	%rsi, -32(%rdx)
	movq	%rdi, -24(%rdx)
	movl	$1, %esi
	movq	%rsi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rbx, -72(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$30, -24(%rdx)
	movq	$900, -32(%rdx)
	movq	$230400, -40(%rdx)
	movq	$28, -48(%rdx)
	movq	$28, -56(%rdx)
	movq	$256, -64(%rdx)
	movq	$31, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	%r13, %rsp
	imulq	$200704, %rbx, %rdi
	orq	$64, %rdi
	movq	%rdi, -112(%rbp)
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	xorl	%eax, %eax
	movq	%r15, %rcx
	movq	3968(%rbp), %r10
	jmp	.LBB0_1352
	.p2align	4, 0x90
.LBB0_1362:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1352:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1363
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1354
	.p2align	4, 0x90
.LBB0_1361:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1354:
	cmpq	$255, %rsi
	jg	.LBB0_1362
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1356
	.p2align	4, 0x90
.LBB0_1360:
	incq	%r8
	addq	$56, %rdi
.LBB0_1356:
	cmpq	$13, %r8
	jg	.LBB0_1361
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1360
	.p2align	4, 0x90
.LBB0_1359:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1359
	jmp	.LBB0_1360
.LBB0_1363:
	xorl	%eax, %eax
	jmp	.LBB0_1364
	.p2align	4, 0x90
.LBB0_1383:
	incq	%rax
	addq	$921600, -72(%rbp)
.LBB0_1364:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1384
	movq	4008(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1366
	.p2align	4, 0x90
.LBB0_1382:
	incq	%rdx
	addq	$9216, -56(%rbp)
.LBB0_1366:
	cmpq	$255, %rdx
	jg	.LBB0_1383
	movq	-72(%rbp), %rcx
	xorl	%r10d, %r10d
	jmp	.LBB0_1368
	.p2align	4, 0x90
.LBB0_1381:
	incq	%r10
	movq	-64(%rbp), %rcx
	addq	$240, %rcx
.LBB0_1368:
	cmpq	$13, %r10
	jg	.LBB0_1382
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1370
	.p2align	4, 0x90
.LBB0_1380:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$8, %rcx
.LBB0_1370:
	cmpq	$13, %r9
	jg	.LBB0_1381
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r12
	movq	-56(%rbp), %r11
	xorl	%r14d, %r14d
	jmp	.LBB0_1372
	.p2align	4, 0x90
.LBB0_1379:
	incq	%r14
	addq	$36, %r11
	movq	-88(%rbp), %r12
	addq	$3600, %r12
.LBB0_1372:
	cmpq	$255, %r14
	jg	.LBB0_1380
	movq	%r12, -88(%rbp)
	movq	%r11, %r13
	xorl	%ebx, %ebx
	jmp	.LBB0_1374
	.p2align	4, 0x90
.LBB0_1378:
	incq	%rbx
	addq	$12, %r13
	addq	$120, %r12
.LBB0_1374:
	cmpq	$2, %rbx
	jg	.LBB0_1379
	xorl	%esi, %esi
	cmpq	$2, %rsi
	jg	.LBB0_1378
	.p2align	4, 0x90
.LBB0_1377:
	movss	(%r12,%rsi,4), %xmm0
	imulq	$50176, %rax, %rcx
	imulq	$196, %rdx, %r8
	addq	%rcx, %r8
	movq	%r10, %rdi
	shlq	$4, %rdi
	subq	%r10, %rdi
	subq	%r10, %rdi
	addq	%r8, %rdi
	addq	%r9, %rdi
	mulss	(%r13,%rsi,4), %xmm0
	addss	(%r15,%rdi,4), %xmm0
	movss	%xmm0, (%r15,%rdi,4)
	incq	%rsi
	cmpq	$2, %rsi
	jle	.LBB0_1377
	jmp	.LBB0_1378
.LBB0_1384:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r15, %rcx
	jmp	.LBB0_1385
	.p2align	4, 0x90
.LBB0_1395:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1385:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1396
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1387
	.p2align	4, 0x90
.LBB0_1394:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1387:
	cmpq	$255, %rsi
	jg	.LBB0_1395
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1389
	.p2align	4, 0x90
.LBB0_1393:
	incq	%r8
	addq	$56, %rdi
.LBB0_1389:
	cmpq	$13, %r8
	jg	.LBB0_1394
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1393
	.p2align	4, 0x90
.LBB0_1392:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1392
	jmp	.LBB0_1393
.LBB0_1396:
	movq	-128(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r13
	addq	$63, %r13
	andq	$-64, %r13
	xorl	%eax, %eax
	movq	%r13, %rcx
	movq	4224(%rbp), %r10
	movq	3840(%rbp), %r11
	jmp	.LBB0_1397
	.p2align	4, 0x90
.LBB0_1407:
	incq	%rax
	addq	$802816, %rcx
.LBB0_1397:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1408
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1399
	.p2align	4, 0x90
.LBB0_1406:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1399:
	cmpq	$1023, %rsi
	jg	.LBB0_1407
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1401
	.p2align	4, 0x90
.LBB0_1405:
	incq	%r8
	addq	$56, %rdi
.LBB0_1401:
	cmpq	$13, %r8
	jg	.LBB0_1406
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1405
	.p2align	4, 0x90
.LBB0_1404:
	movss	(%r11,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1404
	jmp	.LBB0_1405
.LBB0_1408:
	xorl	%eax, %eax
	jmp	.LBB0_1409
	.p2align	4, 0x90
.LBB0_1428:
	incq	%rax
	addq	$200704, %r15
.LBB0_1409:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1429
	movq	3880(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1411
	.p2align	4, 0x90
.LBB0_1427:
	incq	%rdx
	addq	$1024, -72(%rbp)
	movq	4224(%rbp), %r10
.LBB0_1411:
	cmpq	$1023, %rdx
	jg	.LBB0_1428
	movq	%r15, %rcx
	xorl	%ebx, %ebx
	jmp	.LBB0_1413
	.p2align	4, 0x90
.LBB0_1426:
	incq	%rbx
	movq	-96(%rbp), %rcx
	addq	$56, %rcx
.LBB0_1413:
	cmpq	$13, %rbx
	jg	.LBB0_1427
	movq	%rcx, -96(%rbp)
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1415
	.p2align	4, 0x90
.LBB0_1425:
	incq	%r9
	addq	$4, -64(%rbp)
.LBB0_1415:
	cmpq	$13, %r9
	jg	.LBB0_1426
	movq	-64(%rbp), %r14
	movq	-72(%rbp), %r11
	xorl	%ecx, %ecx
	jmp	.LBB0_1417
	.p2align	4, 0x90
.LBB0_1424:
	movq	-56(%rbp), %rcx
	incq	%rcx
	movq	-80(%rbp), %r11
	addq	$4, %r11
	movq	-88(%rbp), %r14
	addq	$784, %r14
.LBB0_1417:
	cmpq	$255, %rcx
	jg	.LBB0_1425
	movq	%rcx, -56(%rbp)
	movq	%r14, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1419
	.p2align	4, 0x90
.LBB0_1423:
	incq	%rsi
	addq	$4, %r11
	addq	$56, %r14
.LBB0_1419:
	testq	%rsi, %rsi
	jg	.LBB0_1424
	xorl	%r8d, %r8d
	testq	%r8, %r8
	jg	.LBB0_1423
	.p2align	4, 0x90
.LBB0_1422:
	movss	(%r14,%r8,4), %xmm0
	imulq	$200704, %rax, %r10
	imulq	$196, %rdx, %rcx
	addq	%r10, %rcx
	movq	%rbx, %rdi
	shlq	$4, %rdi
	subq	%rbx, %rdi
	subq	%rbx, %rdi
	addq	%rcx, %rdi
	addq	%r9, %rdi
	mulss	(%r11,%r8,4), %xmm0
	addss	(%r13,%rdi,4), %xmm0
	movss	%xmm0, (%r13,%rdi,4)
	incq	%r8
	testq	%r8, %r8
	jle	.LBB0_1422
	jmp	.LBB0_1423
.LBB0_1429:
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_1430
	.p2align	4, 0x90
.LBB0_1440:
	incq	%rax
	addq	$802816, %rcx
.LBB0_1430:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1441
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1432
	.p2align	4, 0x90
.LBB0_1439:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1432:
	cmpq	$1023, %rsi
	jg	.LBB0_1440
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1434
	.p2align	4, 0x90
.LBB0_1438:
	incq	%r8
	addq	$56, %rdi
.LBB0_1434:
	cmpq	$13, %r8
	jg	.LBB0_1439
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1438
	.p2align	4, 0x90
.LBB0_1437:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1437
	jmp	.LBB0_1438
.LBB0_1441:
	xorl	%eax, %eax
	jmp	.LBB0_1442
	.p2align	4, 0x90
.LBB0_1461:
	incq	%rax
	addq	$1605632, -104(%rbp)
.LBB0_1442:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1462
	movq	4264(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1444
	.p2align	4, 0x90
.LBB0_1460:
	incq	%rdx
	addq	$2048, -64(%rbp)
.LBB0_1444:
	cmpq	$1023, %rdx
	jg	.LBB0_1461
	movq	-104(%rbp), %rcx
	xorl	%r11d, %r11d
	jmp	.LBB0_1446
	.p2align	4, 0x90
.LBB0_1459:
	incq	%r11
	movq	-72(%rbp), %rcx
	addq	$224, %rcx
.LBB0_1446:
	cmpq	$13, %r11
	jg	.LBB0_1460
	movq	%rcx, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1448
	.p2align	4, 0x90
.LBB0_1458:
	incq	%r9
	movq	-56(%rbp), %rcx
	addq	$8, %rcx
.LBB0_1448:
	cmpq	$13, %r9
	jg	.LBB0_1459
	movq	%rcx, -56(%rbp)
	movq	%rcx, %r14
	movq	-64(%rbp), %r15
	xorl	%ebx, %ebx
	jmp	.LBB0_1450
	.p2align	4, 0x90
.LBB0_1457:
	incq	%rbx
	movq	-80(%rbp), %r15
	addq	$4, %r15
	movq	-88(%rbp), %r14
	addq	$3136, %r14
.LBB0_1450:
	cmpq	$511, %rbx
	jg	.LBB0_1458
	movq	%r14, -88(%rbp)
	movq	%r15, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1452
	.p2align	4, 0x90
.LBB0_1456:
	incq	%rsi
	addq	$4, %r15
	addq	$112, %r14
.LBB0_1452:
	testq	%rsi, %rsi
	jg	.LBB0_1457
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_1456
	.p2align	4, 0x90
.LBB0_1455:
	movss	(%r14,%rcx,4), %xmm0
	imulq	$200704, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%r11, %rdi
	shlq	$4, %rdi
	subq	%r11, %rdi
	subq	%r11, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r15,%rcx,4), %xmm0
	addss	(%r12,%rdi,4), %xmm0
	movss	%xmm0, (%r12,%rdi,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_1455
	jmp	.LBB0_1456
.LBB0_1462:
	xorl	%eax, %eax
	movq	%r13, %rcx
	jmp	.LBB0_1463
	.p2align	4, 0x90
.LBB0_1473:
	incq	%rax
	addq	$802816, %r12
	addq	$802816, %rcx
.LBB0_1463:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1474
	movq	%rcx, %rdx
	movq	%r12, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1465
	.p2align	4, 0x90
.LBB0_1472:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_1465:
	cmpq	$1023, %rdi
	jg	.LBB0_1473
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1467
	.p2align	4, 0x90
.LBB0_1471:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_1467:
	cmpq	$13, %r10
	jg	.LBB0_1472
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_1471
	.p2align	4, 0x90
.LBB0_1470:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_1470
	jmp	.LBB0_1471
.LBB0_1474:
	movq	-128(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_1475
	.p2align	4, 0x90
.LBB0_1485:
	incq	%rbx
	addq	$802816, %rcx
	addq	$802816, %r13
.LBB0_1475:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_1486
	movq	%r13, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1477
	.p2align	4, 0x90
.LBB0_1484:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_1477:
	cmpq	$1023, %rdi
	jg	.LBB0_1485
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1479
	.p2align	4, 0x90
.LBB0_1483:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_1479:
	cmpq	$13, %r10
	jg	.LBB0_1484
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_1483
	.p2align	4, 0x90
.LBB0_1482:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_1482
	jmp	.LBB0_1483
.LBB0_1486:
	movq	%rax, -120(%rbp)
	movq	-112(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -144(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	3712(%rbp), %r10
	jmp	.LBB0_1487
	.p2align	4, 0x90
.LBB0_1497:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1487:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1498
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1489
	.p2align	4, 0x90
.LBB0_1496:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1489:
	cmpq	$255, %rsi
	jg	.LBB0_1497
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1491
	.p2align	4, 0x90
.LBB0_1495:
	incq	%r8
	addq	$56, %rdi
.LBB0_1491:
	cmpq	$13, %r8
	jg	.LBB0_1496
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1495
	.p2align	4, 0x90
.LBB0_1494:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1494
	jmp	.LBB0_1495
.LBB0_1498:
	xorl	%eax, %eax
	movq	-120(%rbp), %rcx
	movq	%rcx, -96(%rbp)
	jmp	.LBB0_1499
	.p2align	4, 0x90
.LBB0_1518:
	incq	%rax
	addq	$802816, -96(%rbp)
.LBB0_1499:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1519
	movq	3752(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1501
	.p2align	4, 0x90
.LBB0_1517:
	incq	%rsi
	addq	$4096, -64(%rbp)
.LBB0_1501:
	cmpq	$255, %rsi
	jg	.LBB0_1518
	movq	-96(%rbp), %r11
	xorl	%edx, %edx
	jmp	.LBB0_1503
	.p2align	4, 0x90
.LBB0_1516:
	incq	%rdx
	movq	-72(%rbp), %r11
	addq	$56, %r11
.LBB0_1503:
	cmpq	$13, %rdx
	jg	.LBB0_1517
	movq	%r11, -72(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_1505
	.p2align	4, 0x90
.LBB0_1515:
	incq	%r10
	movq	-56(%rbp), %r11
	addq	$4, %r11
.LBB0_1505:
	cmpq	$13, %r10
	jg	.LBB0_1516
	movq	%r11, -56(%rbp)
	movq	-64(%rbp), %r14
	xorl	%r13d, %r13d
	jmp	.LBB0_1507
	.p2align	4, 0x90
.LBB0_1514:
	incq	%r13
	movq	-80(%rbp), %r14
	addq	$4, %r14
	movq	-88(%rbp), %r11
	addq	$784, %r11
.LBB0_1507:
	cmpq	$1023, %r13
	jg	.LBB0_1515
	movq	%r11, -88(%rbp)
	movq	%r14, -80(%rbp)
	xorl	%ecx, %ecx
	jmp	.LBB0_1509
	.p2align	4, 0x90
.LBB0_1513:
	incq	%rcx
	addq	$4, %r14
	addq	$56, %r11
.LBB0_1509:
	testq	%rcx, %rcx
	jg	.LBB0_1514
	xorl	%r15d, %r15d
	testq	%r15, %r15
	jg	.LBB0_1513
	.p2align	4, 0x90
.LBB0_1512:
	movss	(%r11,%r15,4), %xmm0
	imulq	$50176, %rax, %rdi
	imulq	$196, %rsi, %r9
	addq	%rdi, %r9
	movq	%rdx, %r8
	shlq	$4, %r8
	subq	%rdx, %r8
	subq	%rdx, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r14,%r15,4), %xmm0
	addss	(%rbx,%r8,4), %xmm0
	movss	%xmm0, (%rbx,%r8,4)
	incq	%r15
	testq	%r15, %r15
	jle	.LBB0_1512
	jmp	.LBB0_1513
.LBB0_1519:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1520
	.p2align	4, 0x90
.LBB0_1530:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1520:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1531
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1522
	.p2align	4, 0x90
.LBB0_1529:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1522:
	cmpq	$255, %rsi
	jg	.LBB0_1530
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1524
	.p2align	4, 0x90
.LBB0_1528:
	incq	%r8
	addq	$56, %rdi
.LBB0_1524:
	cmpq	$13, %r8
	jg	.LBB0_1529
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1528
	.p2align	4, 0x90
.LBB0_1527:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1527
	jmp	.LBB0_1528
.LBB0_1531:
	movq	-48(%rbp), %rdi
	shlq	$18, %rdi
	orq	$64, %rdi
	movq	%rdi, -104(%rbp)
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -96(%rbp)
	jmp	.LBB0_1532
	.p2align	4, 0x90
.LBB0_1542:
	incq	%rcx
	addq	$262144, %rdx
.LBB0_1532:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_1543
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1534
	.p2align	4, 0x90
.LBB0_1541:
	incq	%rdi
	addq	$1024, %rsi
.LBB0_1534:
	cmpq	$255, %rdi
	jg	.LBB0_1542
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_1536
	.p2align	4, 0x90
.LBB0_1540:
	incq	%r9
	addq	$64, %r8
.LBB0_1536:
	cmpq	$15, %r9
	jg	.LBB0_1541
	xorl	%r10d, %r10d
	cmpq	$15, %r10
	jg	.LBB0_1540
	.p2align	4, 0x90
.LBB0_1539:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$15, %r10
	jle	.LBB0_1539
	jmp	.LBB0_1540
.LBB0_1543:
	movq	%rsp, %r13
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$14, %r8d
	movq	%r8, -56(%rdx)
	movl	$256, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-144(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$50176, %edi
	movq	%rdi, -40(%rdx)
	movl	$196, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-96(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$16, -24(%rdx)
	movq	$256, -32(%rdx)
	movq	$65536, -40(%rdx)
	movq	$14, -48(%rdx)
	movq	$14, -56(%rdx)
	movq	$256, -64(%rdx)
	movq	$17, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r13, %rsp
	movq	%rbx, %rcx
	movq	3584(%rbp), %r10
	jmp	.LBB0_1544
	.p2align	4, 0x90
.LBB0_1554:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1544:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1555
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1546
	.p2align	4, 0x90
.LBB0_1553:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1546:
	cmpq	$255, %rsi
	jg	.LBB0_1554
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1548
	.p2align	4, 0x90
.LBB0_1552:
	incq	%r8
	addq	$56, %rdi
.LBB0_1548:
	cmpq	$13, %r8
	jg	.LBB0_1553
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1552
	.p2align	4, 0x90
.LBB0_1551:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1551
	jmp	.LBB0_1552
.LBB0_1555:
	xorl	%eax, %eax
	jmp	.LBB0_1556
	.p2align	4, 0x90
.LBB0_1575:
	incq	%rax
	addq	$262144, -96(%rbp)
.LBB0_1556:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1576
	movq	3624(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1558
	.p2align	4, 0x90
.LBB0_1574:
	incq	%rdx
	addq	$9216, -64(%rbp)
.LBB0_1558:
	cmpq	$255, %rdx
	jg	.LBB0_1575
	movq	-96(%rbp), %rsi
	xorl	%ecx, %ecx
	jmp	.LBB0_1560
	.p2align	4, 0x90
.LBB0_1573:
	incq	%rcx
	movq	-72(%rbp), %rsi
	addq	$64, %rsi
.LBB0_1560:
	cmpq	$13, %rcx
	jg	.LBB0_1574
	movq	%rsi, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1562
	.p2align	4, 0x90
.LBB0_1572:
	incq	%r9
	movq	-56(%rbp), %rsi
	addq	$4, %rsi
.LBB0_1562:
	cmpq	$13, %r9
	jg	.LBB0_1573
	movq	%rsi, -56(%rbp)
	movq	%rsi, %r15
	movq	-64(%rbp), %r11
	xorl	%r13d, %r13d
	jmp	.LBB0_1564
	.p2align	4, 0x90
.LBB0_1571:
	incq	%r13
	movq	-80(%rbp), %r11
	addq	$36, %r11
	movq	-88(%rbp), %r15
	addq	$1024, %r15
.LBB0_1564:
	cmpq	$255, %r13
	jg	.LBB0_1572
	movq	%r15, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1566
	.p2align	4, 0x90
.LBB0_1570:
	incq	%rsi
	addq	$12, %r11
	addq	$64, %r15
.LBB0_1566:
	cmpq	$2, %rsi
	jg	.LBB0_1571
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_1570
	.p2align	4, 0x90
.LBB0_1569:
	movss	(%r15,%r14,4), %xmm0
	imulq	$50176, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%rcx, %rdi
	shlq	$4, %rdi
	subq	%rcx, %rdi
	subq	%rcx, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%rdi,4), %xmm0
	movss	%xmm0, (%rbx,%rdi,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_1569
	jmp	.LBB0_1570
.LBB0_1576:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1577
	.p2align	4, 0x90
.LBB0_1587:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1577:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1588
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1579
	.p2align	4, 0x90
.LBB0_1586:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1579:
	cmpq	$255, %rsi
	jg	.LBB0_1587
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1581
	.p2align	4, 0x90
.LBB0_1585:
	incq	%r8
	addq	$56, %rdi
.LBB0_1581:
	cmpq	$13, %r8
	jg	.LBB0_1586
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1585
	.p2align	4, 0x90
.LBB0_1584:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1584
	jmp	.LBB0_1585
.LBB0_1588:
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_1589
	.p2align	4, 0x90
.LBB0_1599:
	incq	%rax
	addq	$802816, %rcx
.LBB0_1589:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1600
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1591
	.p2align	4, 0x90
.LBB0_1598:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1591:
	cmpq	$1023, %rsi
	jg	.LBB0_1599
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1593
	.p2align	4, 0x90
.LBB0_1597:
	incq	%r8
	addq	$56, %rdi
.LBB0_1593:
	cmpq	$13, %r8
	jg	.LBB0_1598
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1597
	.p2align	4, 0x90
.LBB0_1596:
	movq	3456(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1596
	jmp	.LBB0_1597
.LBB0_1600:
	xorl	%eax, %eax
	jmp	.LBB0_1601
	.p2align	4, 0x90
.LBB0_1620:
	incq	%rax
	addq	$200704, %rbx
.LBB0_1601:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1621
	movq	3496(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1603
	.p2align	4, 0x90
.LBB0_1619:
	incq	%rdx
	addq	$1024, -64(%rbp)
.LBB0_1603:
	cmpq	$1023, %rdx
	jg	.LBB0_1620
	movq	%rbx, %rcx
	xorl	%r11d, %r11d
	jmp	.LBB0_1605
	.p2align	4, 0x90
.LBB0_1618:
	incq	%r11
	movq	-72(%rbp), %rcx
	addq	$56, %rcx
.LBB0_1605:
	cmpq	$13, %r11
	jg	.LBB0_1619
	movq	%rcx, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1607
	.p2align	4, 0x90
.LBB0_1617:
	incq	%r9
	movq	-56(%rbp), %rcx
	addq	$4, %rcx
.LBB0_1607:
	cmpq	$13, %r9
	jg	.LBB0_1618
	movq	%rcx, -56(%rbp)
	movq	%rcx, %r15
	movq	-64(%rbp), %r13
	xorl	%r14d, %r14d
	jmp	.LBB0_1609
	.p2align	4, 0x90
.LBB0_1616:
	incq	%r14
	movq	-80(%rbp), %r13
	addq	$4, %r13
	movq	-88(%rbp), %r15
	addq	$784, %r15
.LBB0_1609:
	cmpq	$255, %r14
	jg	.LBB0_1617
	movq	%r15, -88(%rbp)
	movq	%r13, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1611
	.p2align	4, 0x90
.LBB0_1615:
	incq	%rsi
	addq	$4, %r13
	addq	$56, %r15
.LBB0_1611:
	testq	%rsi, %rsi
	jg	.LBB0_1616
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_1615
	.p2align	4, 0x90
.LBB0_1614:
	movss	(%r15,%rcx,4), %xmm0
	imulq	$200704, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%r11, %rdi
	shlq	$4, %rdi
	subq	%r11, %rdi
	subq	%r11, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r12,%rdi,4), %xmm0
	movss	%xmm0, (%r12,%rdi,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_1614
	jmp	.LBB0_1615
.LBB0_1621:
	xorl	%eax, %eax
	movq	%r12, %rcx
	movq	-120(%rbp), %rbx
	jmp	.LBB0_1622
	.p2align	4, 0x90
.LBB0_1632:
	incq	%rax
	addq	$802816, %rbx
	addq	$802816, %rcx
.LBB0_1622:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1633
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1624
	.p2align	4, 0x90
.LBB0_1631:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_1624:
	cmpq	$1023, %rdi
	jg	.LBB0_1632
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1626
	.p2align	4, 0x90
.LBB0_1630:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_1626:
	cmpq	$13, %r10
	jg	.LBB0_1631
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_1630
	.p2align	4, 0x90
.LBB0_1629:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_1629
	jmp	.LBB0_1630
.LBB0_1633:
	movq	-128(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_1634
	.p2align	4, 0x90
.LBB0_1644:
	incq	%rbx
	addq	$802816, %rcx
	addq	$802816, %r12
.LBB0_1634:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_1645
	movq	%r12, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1636
	.p2align	4, 0x90
.LBB0_1643:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_1636:
	cmpq	$1023, %rdi
	jg	.LBB0_1644
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1638
	.p2align	4, 0x90
.LBB0_1642:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_1638:
	cmpq	$13, %r10
	jg	.LBB0_1643
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_1642
	.p2align	4, 0x90
.LBB0_1641:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_1641
	jmp	.LBB0_1642
.LBB0_1645:
	movq	%rax, -120(%rbp)
	movq	-112(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -144(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	3328(%rbp), %r10
	jmp	.LBB0_1646
	.p2align	4, 0x90
.LBB0_1656:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1646:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1657
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1648
	.p2align	4, 0x90
.LBB0_1655:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1648:
	cmpq	$255, %rsi
	jg	.LBB0_1656
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1650
	.p2align	4, 0x90
.LBB0_1654:
	incq	%r8
	addq	$56, %rdi
.LBB0_1650:
	cmpq	$13, %r8
	jg	.LBB0_1655
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1654
	.p2align	4, 0x90
.LBB0_1653:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1653
	jmp	.LBB0_1654
.LBB0_1657:
	xorl	%eax, %eax
	movq	-120(%rbp), %rcx
	movq	%rcx, -96(%rbp)
	jmp	.LBB0_1658
	.p2align	4, 0x90
.LBB0_1677:
	incq	%rax
	addq	$802816, -96(%rbp)
.LBB0_1658:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1678
	movq	3368(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1660
	.p2align	4, 0x90
.LBB0_1676:
	incq	%rsi
	addq	$4096, -64(%rbp)
.LBB0_1660:
	cmpq	$255, %rsi
	jg	.LBB0_1677
	movq	-96(%rbp), %r11
	xorl	%edx, %edx
	jmp	.LBB0_1662
	.p2align	4, 0x90
.LBB0_1675:
	incq	%rdx
	movq	-72(%rbp), %r11
	addq	$56, %r11
.LBB0_1662:
	cmpq	$13, %rdx
	jg	.LBB0_1676
	movq	%r11, -72(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_1664
	.p2align	4, 0x90
.LBB0_1674:
	incq	%r10
	movq	-56(%rbp), %r11
	addq	$4, %r11
.LBB0_1664:
	cmpq	$13, %r10
	jg	.LBB0_1675
	movq	%r11, -56(%rbp)
	movq	-64(%rbp), %r14
	xorl	%r12d, %r12d
	jmp	.LBB0_1666
	.p2align	4, 0x90
.LBB0_1673:
	incq	%r12
	movq	-80(%rbp), %r14
	addq	$4, %r14
	movq	-88(%rbp), %r11
	addq	$784, %r11
.LBB0_1666:
	cmpq	$1023, %r12
	jg	.LBB0_1674
	movq	%r11, -88(%rbp)
	movq	%r14, -80(%rbp)
	xorl	%ecx, %ecx
	jmp	.LBB0_1668
	.p2align	4, 0x90
.LBB0_1672:
	incq	%rcx
	addq	$4, %r14
	addq	$56, %r11
.LBB0_1668:
	testq	%rcx, %rcx
	jg	.LBB0_1673
	xorl	%r13d, %r13d
	testq	%r13, %r13
	jg	.LBB0_1672
	.p2align	4, 0x90
.LBB0_1671:
	movss	(%r11,%r13,4), %xmm0
	imulq	$50176, %rax, %rdi
	imulq	$196, %rsi, %r9
	addq	%rdi, %r9
	movq	%rdx, %r8
	shlq	$4, %r8
	subq	%rdx, %r8
	subq	%rdx, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r14,%r13,4), %xmm0
	addss	(%rbx,%r8,4), %xmm0
	movss	%xmm0, (%rbx,%r8,4)
	incq	%r13
	testq	%r13, %r13
	jle	.LBB0_1671
	jmp	.LBB0_1672
.LBB0_1678:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1679
	.p2align	4, 0x90
.LBB0_1689:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1679:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1690
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1681
	.p2align	4, 0x90
.LBB0_1688:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1681:
	cmpq	$255, %rsi
	jg	.LBB0_1689
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1683
	.p2align	4, 0x90
.LBB0_1687:
	incq	%r8
	addq	$56, %rdi
.LBB0_1683:
	cmpq	$13, %r8
	jg	.LBB0_1688
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1687
	.p2align	4, 0x90
.LBB0_1686:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1686
	jmp	.LBB0_1687
.LBB0_1690:
	movq	-104(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -96(%rbp)
	jmp	.LBB0_1691
	.p2align	4, 0x90
.LBB0_1701:
	incq	%rcx
	addq	$262144, %rdx
.LBB0_1691:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_1702
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1693
	.p2align	4, 0x90
.LBB0_1700:
	incq	%rdi
	addq	$1024, %rsi
.LBB0_1693:
	cmpq	$255, %rdi
	jg	.LBB0_1701
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_1695
	.p2align	4, 0x90
.LBB0_1699:
	incq	%r9
	addq	$64, %r8
.LBB0_1695:
	cmpq	$15, %r9
	jg	.LBB0_1700
	xorl	%r10d, %r10d
	cmpq	$15, %r10
	jg	.LBB0_1699
	.p2align	4, 0x90
.LBB0_1698:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$15, %r10
	jle	.LBB0_1698
	jmp	.LBB0_1699
.LBB0_1702:
	movq	%rsp, %r12
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$14, %r8d
	movq	%r8, -56(%rdx)
	movl	$256, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-144(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$50176, %edi
	movq	%rdi, -40(%rdx)
	movl	$196, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-96(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$16, -24(%rdx)
	movq	$256, -32(%rdx)
	movq	$65536, -40(%rdx)
	movq	$14, -48(%rdx)
	movq	$14, -56(%rdx)
	movq	$256, -64(%rdx)
	movq	$17, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r12, %rsp
	movq	%rbx, %rcx
	movq	3200(%rbp), %r10
	jmp	.LBB0_1703
	.p2align	4, 0x90
.LBB0_1713:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1703:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1714
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1705
	.p2align	4, 0x90
.LBB0_1712:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1705:
	cmpq	$255, %rsi
	jg	.LBB0_1713
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1707
	.p2align	4, 0x90
.LBB0_1711:
	incq	%r8
	addq	$56, %rdi
.LBB0_1707:
	cmpq	$13, %r8
	jg	.LBB0_1712
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1711
	.p2align	4, 0x90
.LBB0_1710:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1710
	jmp	.LBB0_1711
.LBB0_1714:
	xorl	%eax, %eax
	jmp	.LBB0_1715
	.p2align	4, 0x90
.LBB0_1734:
	incq	%rax
	addq	$262144, -96(%rbp)
.LBB0_1715:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1735
	movq	3240(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1717
	.p2align	4, 0x90
.LBB0_1733:
	incq	%rdx
	addq	$9216, -64(%rbp)
.LBB0_1717:
	cmpq	$255, %rdx
	jg	.LBB0_1734
	movq	-96(%rbp), %rsi
	xorl	%ecx, %ecx
	jmp	.LBB0_1719
	.p2align	4, 0x90
.LBB0_1732:
	incq	%rcx
	movq	-72(%rbp), %rsi
	addq	$64, %rsi
.LBB0_1719:
	cmpq	$13, %rcx
	jg	.LBB0_1733
	movq	%rsi, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1721
	.p2align	4, 0x90
.LBB0_1731:
	incq	%r9
	movq	-56(%rbp), %rsi
	addq	$4, %rsi
.LBB0_1721:
	cmpq	$13, %r9
	jg	.LBB0_1732
	movq	%rsi, -56(%rbp)
	movq	%rsi, %r13
	movq	-64(%rbp), %r11
	xorl	%r12d, %r12d
	jmp	.LBB0_1723
	.p2align	4, 0x90
.LBB0_1730:
	incq	%r12
	movq	-80(%rbp), %r11
	addq	$36, %r11
	movq	-88(%rbp), %r13
	addq	$1024, %r13
.LBB0_1723:
	cmpq	$255, %r12
	jg	.LBB0_1731
	movq	%r13, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1725
	.p2align	4, 0x90
.LBB0_1729:
	incq	%rsi
	addq	$12, %r11
	addq	$64, %r13
.LBB0_1725:
	cmpq	$2, %rsi
	jg	.LBB0_1730
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_1729
	.p2align	4, 0x90
.LBB0_1728:
	movss	(%r13,%r14,4), %xmm0
	imulq	$50176, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%rcx, %rdi
	shlq	$4, %rdi
	subq	%rcx, %rdi
	subq	%rcx, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%rdi,4), %xmm0
	movss	%xmm0, (%rbx,%rdi,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_1728
	jmp	.LBB0_1729
.LBB0_1735:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1736
	.p2align	4, 0x90
.LBB0_1746:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1736:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1747
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1738
	.p2align	4, 0x90
.LBB0_1745:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1738:
	cmpq	$255, %rsi
	jg	.LBB0_1746
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1740
	.p2align	4, 0x90
.LBB0_1744:
	incq	%r8
	addq	$56, %rdi
.LBB0_1740:
	cmpq	$13, %r8
	jg	.LBB0_1745
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1744
	.p2align	4, 0x90
.LBB0_1743:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1743
	jmp	.LBB0_1744
.LBB0_1747:
	xorl	%eax, %eax
	movq	%r15, %rcx
	jmp	.LBB0_1748
	.p2align	4, 0x90
.LBB0_1758:
	incq	%rax
	addq	$802816, %rcx
.LBB0_1748:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1759
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1750
	.p2align	4, 0x90
.LBB0_1757:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1750:
	cmpq	$1023, %rsi
	jg	.LBB0_1758
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1752
	.p2align	4, 0x90
.LBB0_1756:
	incq	%r8
	addq	$56, %rdi
.LBB0_1752:
	cmpq	$13, %r8
	jg	.LBB0_1757
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1756
	.p2align	4, 0x90
.LBB0_1755:
	movq	1496(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1755
	jmp	.LBB0_1756
.LBB0_1759:
	xorl	%eax, %eax
	jmp	.LBB0_1760
	.p2align	4, 0x90
.LBB0_1779:
	incq	%rax
	addq	$200704, %rbx
.LBB0_1760:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1780
	movq	1664(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1762
	.p2align	4, 0x90
.LBB0_1778:
	incq	%rdx
	addq	$1024, -64(%rbp)
.LBB0_1762:
	cmpq	$1023, %rdx
	jg	.LBB0_1779
	movq	%rbx, %rcx
	xorl	%r11d, %r11d
	jmp	.LBB0_1764
	.p2align	4, 0x90
.LBB0_1777:
	incq	%r11
	movq	-72(%rbp), %rcx
	addq	$56, %rcx
.LBB0_1764:
	cmpq	$13, %r11
	jg	.LBB0_1778
	movq	%rcx, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1766
	.p2align	4, 0x90
.LBB0_1776:
	incq	%r9
	movq	-56(%rbp), %rcx
	addq	$4, %rcx
.LBB0_1766:
	cmpq	$13, %r9
	jg	.LBB0_1777
	movq	%rcx, -56(%rbp)
	movq	%rcx, %r12
	movq	-64(%rbp), %r13
	xorl	%r14d, %r14d
	jmp	.LBB0_1768
	.p2align	4, 0x90
.LBB0_1775:
	incq	%r14
	movq	-80(%rbp), %r13
	addq	$4, %r13
	movq	-88(%rbp), %r12
	addq	$784, %r12
.LBB0_1768:
	cmpq	$255, %r14
	jg	.LBB0_1776
	movq	%r12, -88(%rbp)
	movq	%r13, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1770
	.p2align	4, 0x90
.LBB0_1774:
	incq	%rsi
	addq	$4, %r13
	addq	$56, %r12
.LBB0_1770:
	testq	%rsi, %rsi
	jg	.LBB0_1775
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_1774
	.p2align	4, 0x90
.LBB0_1773:
	movss	(%r12,%rcx,4), %xmm0
	imulq	$200704, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%r11, %rdi
	shlq	$4, %rdi
	subq	%r11, %rdi
	subq	%r11, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r15,%rdi,4), %xmm0
	movss	%xmm0, (%r15,%rdi,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_1773
	jmp	.LBB0_1774
.LBB0_1780:
	xorl	%eax, %eax
	movq	%r15, %rcx
	movq	-120(%rbp), %rbx
	jmp	.LBB0_1781
	.p2align	4, 0x90
.LBB0_1791:
	incq	%rax
	addq	$802816, %rbx
	addq	$802816, %rcx
.LBB0_1781:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1792
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1783
	.p2align	4, 0x90
.LBB0_1790:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_1783:
	cmpq	$1023, %rdi
	jg	.LBB0_1791
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1785
	.p2align	4, 0x90
.LBB0_1789:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_1785:
	cmpq	$13, %r10
	jg	.LBB0_1790
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_1789
	.p2align	4, 0x90
.LBB0_1788:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_1788
	jmp	.LBB0_1789
.LBB0_1792:
	movq	-128(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_1793
	.p2align	4, 0x90
.LBB0_1803:
	incq	%rbx
	addq	$802816, %rcx
	addq	$802816, %r15
.LBB0_1793:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_1804
	movq	%r15, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1795
	.p2align	4, 0x90
.LBB0_1802:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_1795:
	cmpq	$1023, %rdi
	jg	.LBB0_1803
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1797
	.p2align	4, 0x90
.LBB0_1801:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_1797:
	cmpq	$13, %r10
	jg	.LBB0_1802
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_1801
	.p2align	4, 0x90
.LBB0_1800:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_1800
	jmp	.LBB0_1801
.LBB0_1804:
	movq	%rax, -120(%rbp)
	movq	-112(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -144(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	1312(%rbp), %r10
	jmp	.LBB0_1805
	.p2align	4, 0x90
.LBB0_1815:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1805:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1816
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1807
	.p2align	4, 0x90
.LBB0_1814:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1807:
	cmpq	$255, %rsi
	jg	.LBB0_1815
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1809
	.p2align	4, 0x90
.LBB0_1813:
	incq	%r8
	addq	$56, %rdi
.LBB0_1809:
	cmpq	$13, %r8
	jg	.LBB0_1814
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1813
	.p2align	4, 0x90
.LBB0_1812:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1812
	jmp	.LBB0_1813
.LBB0_1816:
	xorl	%eax, %eax
	movq	-120(%rbp), %rcx
	movq	%rcx, -96(%rbp)
	jmp	.LBB0_1817
	.p2align	4, 0x90
.LBB0_1836:
	incq	%rax
	addq	$802816, -96(%rbp)
.LBB0_1817:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1837
	movq	1352(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1819
	.p2align	4, 0x90
.LBB0_1835:
	incq	%rsi
	addq	$4096, -64(%rbp)
.LBB0_1819:
	cmpq	$255, %rsi
	jg	.LBB0_1836
	movq	-96(%rbp), %r11
	xorl	%edx, %edx
	jmp	.LBB0_1821
	.p2align	4, 0x90
.LBB0_1834:
	incq	%rdx
	movq	-72(%rbp), %r11
	addq	$56, %r11
.LBB0_1821:
	cmpq	$13, %rdx
	jg	.LBB0_1835
	movq	%r11, -72(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_1823
	.p2align	4, 0x90
.LBB0_1833:
	incq	%r10
	movq	-56(%rbp), %r11
	addq	$4, %r11
.LBB0_1823:
	cmpq	$13, %r10
	jg	.LBB0_1834
	movq	%r11, -56(%rbp)
	movq	-64(%rbp), %r14
	xorl	%r15d, %r15d
	jmp	.LBB0_1825
	.p2align	4, 0x90
.LBB0_1832:
	incq	%r15
	movq	-80(%rbp), %r14
	addq	$4, %r14
	movq	-88(%rbp), %r11
	addq	$784, %r11
.LBB0_1825:
	cmpq	$1023, %r15
	jg	.LBB0_1833
	movq	%r11, -88(%rbp)
	movq	%r14, -80(%rbp)
	xorl	%ecx, %ecx
	jmp	.LBB0_1827
	.p2align	4, 0x90
.LBB0_1831:
	incq	%rcx
	addq	$4, %r14
	addq	$56, %r11
.LBB0_1827:
	testq	%rcx, %rcx
	jg	.LBB0_1832
	xorl	%r13d, %r13d
	testq	%r13, %r13
	jg	.LBB0_1831
	.p2align	4, 0x90
.LBB0_1830:
	movss	(%r11,%r13,4), %xmm0
	imulq	$50176, %rax, %rdi
	imulq	$196, %rsi, %r9
	addq	%rdi, %r9
	movq	%rdx, %r8
	shlq	$4, %r8
	subq	%rdx, %r8
	subq	%rdx, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r14,%r13,4), %xmm0
	addss	(%rbx,%r8,4), %xmm0
	movss	%xmm0, (%rbx,%r8,4)
	incq	%r13
	testq	%r13, %r13
	jle	.LBB0_1830
	jmp	.LBB0_1831
.LBB0_1837:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1838
	.p2align	4, 0x90
.LBB0_1848:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1838:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1849
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1840
	.p2align	4, 0x90
.LBB0_1847:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1840:
	cmpq	$255, %rsi
	jg	.LBB0_1848
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1842
	.p2align	4, 0x90
.LBB0_1846:
	incq	%r8
	addq	$56, %rdi
.LBB0_1842:
	cmpq	$13, %r8
	jg	.LBB0_1847
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1846
	.p2align	4, 0x90
.LBB0_1845:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1845
	jmp	.LBB0_1846
.LBB0_1849:
	movq	-104(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -96(%rbp)
	jmp	.LBB0_1850
	.p2align	4, 0x90
.LBB0_1860:
	incq	%rcx
	addq	$262144, %rdx
.LBB0_1850:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_1861
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1852
	.p2align	4, 0x90
.LBB0_1859:
	incq	%rdi
	addq	$1024, %rsi
.LBB0_1852:
	cmpq	$255, %rdi
	jg	.LBB0_1860
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_1854
	.p2align	4, 0x90
.LBB0_1858:
	incq	%r9
	addq	$64, %r8
.LBB0_1854:
	cmpq	$15, %r9
	jg	.LBB0_1859
	xorl	%r10d, %r10d
	cmpq	$15, %r10
	jg	.LBB0_1858
	.p2align	4, 0x90
.LBB0_1857:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$15, %r10
	jle	.LBB0_1857
	jmp	.LBB0_1858
.LBB0_1861:
	movq	%rsp, %r15
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$14, %r8d
	movq	%r8, -56(%rdx)
	movl	$256, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-144(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$50176, %edi
	movq	%rdi, -40(%rdx)
	movl	$196, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-96(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$16, -24(%rdx)
	movq	$256, -32(%rdx)
	movq	$65536, -40(%rdx)
	movq	$14, -48(%rdx)
	movq	$14, -56(%rdx)
	movq	$256, -64(%rdx)
	movq	$17, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r15, %rsp
	movq	%rbx, %rcx
	movq	1184(%rbp), %r10
	jmp	.LBB0_1862
	.p2align	4, 0x90
.LBB0_1872:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1862:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1873
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1864
	.p2align	4, 0x90
.LBB0_1871:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1864:
	cmpq	$255, %rsi
	jg	.LBB0_1872
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1866
	.p2align	4, 0x90
.LBB0_1870:
	incq	%r8
	addq	$56, %rdi
.LBB0_1866:
	cmpq	$13, %r8
	jg	.LBB0_1871
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1870
	.p2align	4, 0x90
.LBB0_1869:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1869
	jmp	.LBB0_1870
.LBB0_1873:
	xorl	%eax, %eax
	jmp	.LBB0_1874
	.p2align	4, 0x90
.LBB0_1893:
	incq	%rax
	addq	$262144, -96(%rbp)
.LBB0_1874:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1894
	movq	1224(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1876
	.p2align	4, 0x90
.LBB0_1892:
	incq	%rdx
	addq	$9216, -64(%rbp)
.LBB0_1876:
	cmpq	$255, %rdx
	jg	.LBB0_1893
	movq	-96(%rbp), %rsi
	xorl	%ecx, %ecx
	jmp	.LBB0_1878
	.p2align	4, 0x90
.LBB0_1891:
	incq	%rcx
	movq	-72(%rbp), %rsi
	addq	$64, %rsi
.LBB0_1878:
	cmpq	$13, %rcx
	jg	.LBB0_1892
	movq	%rsi, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1880
	.p2align	4, 0x90
.LBB0_1890:
	incq	%r9
	movq	-56(%rbp), %rsi
	addq	$4, %rsi
.LBB0_1880:
	cmpq	$13, %r9
	jg	.LBB0_1891
	movq	%rsi, -56(%rbp)
	movq	%rsi, %r13
	movq	-64(%rbp), %r11
	xorl	%r15d, %r15d
	jmp	.LBB0_1882
	.p2align	4, 0x90
.LBB0_1889:
	incq	%r15
	movq	-80(%rbp), %r11
	addq	$36, %r11
	movq	-88(%rbp), %r13
	addq	$1024, %r13
.LBB0_1882:
	cmpq	$255, %r15
	jg	.LBB0_1890
	movq	%r13, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1884
	.p2align	4, 0x90
.LBB0_1888:
	incq	%rsi
	addq	$12, %r11
	addq	$64, %r13
.LBB0_1884:
	cmpq	$2, %rsi
	jg	.LBB0_1889
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_1888
	.p2align	4, 0x90
.LBB0_1887:
	movss	(%r13,%r14,4), %xmm0
	imulq	$50176, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%rcx, %rdi
	shlq	$4, %rdi
	subq	%rcx, %rdi
	subq	%rcx, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%rdi,4), %xmm0
	movss	%xmm0, (%rbx,%rdi,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_1887
	jmp	.LBB0_1888
.LBB0_1894:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1895
	.p2align	4, 0x90
.LBB0_1905:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1895:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1906
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1897
	.p2align	4, 0x90
.LBB0_1904:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1897:
	cmpq	$255, %rsi
	jg	.LBB0_1905
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1899
	.p2align	4, 0x90
.LBB0_1903:
	incq	%r8
	addq	$56, %rdi
.LBB0_1899:
	cmpq	$13, %r8
	jg	.LBB0_1904
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1903
	.p2align	4, 0x90
.LBB0_1902:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1902
	jmp	.LBB0_1903
.LBB0_1906:
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_1907
	.p2align	4, 0x90
.LBB0_1917:
	incq	%rax
	addq	$802816, %rcx
.LBB0_1907:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1918
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1909
	.p2align	4, 0x90
.LBB0_1916:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1909:
	cmpq	$1023, %rsi
	jg	.LBB0_1917
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1911
	.p2align	4, 0x90
.LBB0_1915:
	incq	%r8
	addq	$56, %rdi
.LBB0_1911:
	cmpq	$13, %r8
	jg	.LBB0_1916
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1915
	.p2align	4, 0x90
.LBB0_1914:
	movq	1056(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1914
	jmp	.LBB0_1915
.LBB0_1918:
	xorl	%eax, %eax
	jmp	.LBB0_1919
	.p2align	4, 0x90
.LBB0_1938:
	incq	%rax
	addq	$200704, %rbx
.LBB0_1919:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1939
	movq	1096(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_1921
	.p2align	4, 0x90
.LBB0_1937:
	incq	%rdx
	addq	$1024, -64(%rbp)
.LBB0_1921:
	cmpq	$1023, %rdx
	jg	.LBB0_1938
	movq	%rbx, %rcx
	xorl	%r11d, %r11d
	jmp	.LBB0_1923
	.p2align	4, 0x90
.LBB0_1936:
	incq	%r11
	movq	-72(%rbp), %rcx
	addq	$56, %rcx
.LBB0_1923:
	cmpq	$13, %r11
	jg	.LBB0_1937
	movq	%rcx, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_1925
	.p2align	4, 0x90
.LBB0_1935:
	incq	%r9
	movq	-56(%rbp), %rcx
	addq	$4, %rcx
.LBB0_1925:
	cmpq	$13, %r9
	jg	.LBB0_1936
	movq	%rcx, -56(%rbp)
	movq	%rcx, %r15
	movq	-64(%rbp), %r13
	xorl	%r14d, %r14d
	jmp	.LBB0_1927
	.p2align	4, 0x90
.LBB0_1934:
	incq	%r14
	movq	-80(%rbp), %r13
	addq	$4, %r13
	movq	-88(%rbp), %r15
	addq	$784, %r15
.LBB0_1927:
	cmpq	$255, %r14
	jg	.LBB0_1935
	movq	%r15, -88(%rbp)
	movq	%r13, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1929
	.p2align	4, 0x90
.LBB0_1933:
	incq	%rsi
	addq	$4, %r13
	addq	$56, %r15
.LBB0_1929:
	testq	%rsi, %rsi
	jg	.LBB0_1934
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_1933
	.p2align	4, 0x90
.LBB0_1932:
	movss	(%r15,%rcx,4), %xmm0
	imulq	$200704, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%r11, %rdi
	shlq	$4, %rdi
	subq	%r11, %rdi
	subq	%r11, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r12,%rdi,4), %xmm0
	movss	%xmm0, (%r12,%rdi,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_1932
	jmp	.LBB0_1933
.LBB0_1939:
	xorl	%eax, %eax
	movq	%r12, %rcx
	movq	-120(%rbp), %rbx
	jmp	.LBB0_1940
	.p2align	4, 0x90
.LBB0_1950:
	incq	%rax
	addq	$802816, %rbx
	addq	$802816, %rcx
.LBB0_1940:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1951
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1942
	.p2align	4, 0x90
.LBB0_1949:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_1942:
	cmpq	$1023, %rdi
	jg	.LBB0_1950
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1944
	.p2align	4, 0x90
.LBB0_1948:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_1944:
	cmpq	$13, %r10
	jg	.LBB0_1949
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_1948
	.p2align	4, 0x90
.LBB0_1947:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_1947
	jmp	.LBB0_1948
.LBB0_1951:
	movq	-128(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_1952
	.p2align	4, 0x90
.LBB0_1962:
	incq	%rbx
	addq	$802816, %rcx
	addq	$802816, %r12
.LBB0_1952:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_1963
	movq	%r12, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_1954
	.p2align	4, 0x90
.LBB0_1961:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_1954:
	cmpq	$1023, %rdi
	jg	.LBB0_1962
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_1956
	.p2align	4, 0x90
.LBB0_1960:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_1956:
	cmpq	$13, %r10
	jg	.LBB0_1961
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_1960
	.p2align	4, 0x90
.LBB0_1959:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_1959
	jmp	.LBB0_1960
.LBB0_1963:
	movq	%rax, -120(%rbp)
	movq	-112(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -144(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	928(%rbp), %r10
	jmp	.LBB0_1964
	.p2align	4, 0x90
.LBB0_1974:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1964:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1975
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1966
	.p2align	4, 0x90
.LBB0_1973:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1966:
	cmpq	$255, %rsi
	jg	.LBB0_1974
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_1968
	.p2align	4, 0x90
.LBB0_1972:
	incq	%r8
	addq	$56, %rdi
.LBB0_1968:
	cmpq	$13, %r8
	jg	.LBB0_1973
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_1972
	.p2align	4, 0x90
.LBB0_1971:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_1971
	jmp	.LBB0_1972
.LBB0_1975:
	xorl	%eax, %eax
	movq	-120(%rbp), %rcx
	movq	%rcx, -96(%rbp)
	jmp	.LBB0_1976
	.p2align	4, 0x90
.LBB0_1995:
	incq	%rax
	addq	$802816, -96(%rbp)
.LBB0_1976:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_1996
	movq	968(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_1978
	.p2align	4, 0x90
.LBB0_1994:
	incq	%rsi
	addq	$4096, -64(%rbp)
.LBB0_1978:
	cmpq	$255, %rsi
	jg	.LBB0_1995
	movq	-96(%rbp), %r11
	xorl	%edx, %edx
	jmp	.LBB0_1980
	.p2align	4, 0x90
.LBB0_1993:
	incq	%rdx
	movq	-72(%rbp), %r11
	addq	$56, %r11
.LBB0_1980:
	cmpq	$13, %rdx
	jg	.LBB0_1994
	movq	%r11, -72(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_1982
	.p2align	4, 0x90
.LBB0_1992:
	incq	%r10
	movq	-56(%rbp), %r11
	addq	$4, %r11
.LBB0_1982:
	cmpq	$13, %r10
	jg	.LBB0_1993
	movq	%r11, -56(%rbp)
	movq	-64(%rbp), %r14
	xorl	%r12d, %r12d
	jmp	.LBB0_1984
	.p2align	4, 0x90
.LBB0_1991:
	incq	%r12
	movq	-80(%rbp), %r14
	addq	$4, %r14
	movq	-88(%rbp), %r11
	addq	$784, %r11
.LBB0_1984:
	cmpq	$1023, %r12
	jg	.LBB0_1992
	movq	%r11, -88(%rbp)
	movq	%r14, -80(%rbp)
	xorl	%ecx, %ecx
	jmp	.LBB0_1986
	.p2align	4, 0x90
.LBB0_1990:
	incq	%rcx
	addq	$4, %r14
	addq	$56, %r11
.LBB0_1986:
	testq	%rcx, %rcx
	jg	.LBB0_1991
	xorl	%r13d, %r13d
	testq	%r13, %r13
	jg	.LBB0_1990
	.p2align	4, 0x90
.LBB0_1989:
	movss	(%r11,%r13,4), %xmm0
	imulq	$50176, %rax, %rdi
	imulq	$196, %rsi, %r9
	addq	%rdi, %r9
	movq	%rdx, %r8
	shlq	$4, %r8
	subq	%rdx, %r8
	subq	%rdx, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r14,%r13,4), %xmm0
	addss	(%rbx,%r8,4), %xmm0
	movss	%xmm0, (%rbx,%r8,4)
	incq	%r13
	testq	%r13, %r13
	jle	.LBB0_1989
	jmp	.LBB0_1990
.LBB0_1996:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_1997
	.p2align	4, 0x90
.LBB0_2007:
	incq	%rax
	addq	$200704, %rcx
.LBB0_1997:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2008
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_1999
	.p2align	4, 0x90
.LBB0_2006:
	incq	%rsi
	addq	$784, %rdx
.LBB0_1999:
	cmpq	$255, %rsi
	jg	.LBB0_2007
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2001
	.p2align	4, 0x90
.LBB0_2005:
	incq	%r8
	addq	$56, %rdi
.LBB0_2001:
	cmpq	$13, %r8
	jg	.LBB0_2006
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2005
	.p2align	4, 0x90
.LBB0_2004:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2004
	jmp	.LBB0_2005
.LBB0_2008:
	movq	-104(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -96(%rbp)
	jmp	.LBB0_2009
	.p2align	4, 0x90
.LBB0_2019:
	incq	%rcx
	addq	$262144, %rdx
.LBB0_2009:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2020
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2011
	.p2align	4, 0x90
.LBB0_2018:
	incq	%rdi
	addq	$1024, %rsi
.LBB0_2011:
	cmpq	$255, %rdi
	jg	.LBB0_2019
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_2013
	.p2align	4, 0x90
.LBB0_2017:
	incq	%r9
	addq	$64, %r8
.LBB0_2013:
	cmpq	$15, %r9
	jg	.LBB0_2018
	xorl	%r10d, %r10d
	cmpq	$15, %r10
	jg	.LBB0_2017
	.p2align	4, 0x90
.LBB0_2016:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$15, %r10
	jle	.LBB0_2016
	jmp	.LBB0_2017
.LBB0_2020:
	movq	%rsp, %r12
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$14, %r8d
	movq	%r8, -56(%rdx)
	movl	$256, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-144(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$50176, %edi
	movq	%rdi, -40(%rdx)
	movl	$196, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-96(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$16, -24(%rdx)
	movq	$256, -32(%rdx)
	movq	$65536, -40(%rdx)
	movq	$14, -48(%rdx)
	movq	$14, -56(%rdx)
	movq	$256, -64(%rdx)
	movq	$17, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r12, %rsp
	movq	%rbx, %rcx
	movq	800(%rbp), %r10
	jmp	.LBB0_2021
	.p2align	4, 0x90
.LBB0_2031:
	incq	%rax
	addq	$200704, %rcx
.LBB0_2021:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2032
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2023
	.p2align	4, 0x90
.LBB0_2030:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2023:
	cmpq	$255, %rsi
	jg	.LBB0_2031
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2025
	.p2align	4, 0x90
.LBB0_2029:
	incq	%r8
	addq	$56, %rdi
.LBB0_2025:
	cmpq	$13, %r8
	jg	.LBB0_2030
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2029
	.p2align	4, 0x90
.LBB0_2028:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2028
	jmp	.LBB0_2029
.LBB0_2032:
	xorl	%eax, %eax
	jmp	.LBB0_2033
	.p2align	4, 0x90
.LBB0_2052:
	incq	%rax
	addq	$262144, -96(%rbp)
.LBB0_2033:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2053
	movq	840(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_2035
	.p2align	4, 0x90
.LBB0_2051:
	incq	%rdx
	addq	$9216, -64(%rbp)
.LBB0_2035:
	cmpq	$255, %rdx
	jg	.LBB0_2052
	movq	-96(%rbp), %rsi
	xorl	%ecx, %ecx
	jmp	.LBB0_2037
	.p2align	4, 0x90
.LBB0_2050:
	incq	%rcx
	movq	-72(%rbp), %rsi
	addq	$64, %rsi
.LBB0_2037:
	cmpq	$13, %rcx
	jg	.LBB0_2051
	movq	%rsi, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_2039
	.p2align	4, 0x90
.LBB0_2049:
	incq	%r9
	movq	-56(%rbp), %rsi
	addq	$4, %rsi
.LBB0_2039:
	cmpq	$13, %r9
	jg	.LBB0_2050
	movq	%rsi, -56(%rbp)
	movq	%rsi, %r13
	movq	-64(%rbp), %r11
	xorl	%r12d, %r12d
	jmp	.LBB0_2041
	.p2align	4, 0x90
.LBB0_2048:
	incq	%r12
	movq	-80(%rbp), %r11
	addq	$36, %r11
	movq	-88(%rbp), %r13
	addq	$1024, %r13
.LBB0_2041:
	cmpq	$255, %r12
	jg	.LBB0_2049
	movq	%r13, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2043
	.p2align	4, 0x90
.LBB0_2047:
	incq	%rsi
	addq	$12, %r11
	addq	$64, %r13
.LBB0_2043:
	cmpq	$2, %rsi
	jg	.LBB0_2048
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_2047
	.p2align	4, 0x90
.LBB0_2046:
	movss	(%r13,%r14,4), %xmm0
	imulq	$50176, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%rcx, %rdi
	shlq	$4, %rdi
	subq	%rcx, %rdi
	subq	%rcx, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%rdi,4), %xmm0
	movss	%xmm0, (%rbx,%rdi,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_2046
	jmp	.LBB0_2047
.LBB0_2053:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_2054
	.p2align	4, 0x90
.LBB0_2064:
	incq	%rax
	addq	$200704, %rcx
.LBB0_2054:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2065
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2056
	.p2align	4, 0x90
.LBB0_2063:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2056:
	cmpq	$255, %rsi
	jg	.LBB0_2064
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2058
	.p2align	4, 0x90
.LBB0_2062:
	incq	%r8
	addq	$56, %rdi
.LBB0_2058:
	cmpq	$13, %r8
	jg	.LBB0_2063
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2062
	.p2align	4, 0x90
.LBB0_2061:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2061
	jmp	.LBB0_2062
.LBB0_2065:
	xorl	%eax, %eax
	movq	%r15, %rcx
	jmp	.LBB0_2066
	.p2align	4, 0x90
.LBB0_2076:
	incq	%rax
	addq	$802816, %rcx
.LBB0_2066:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2077
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2068
	.p2align	4, 0x90
.LBB0_2075:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2068:
	cmpq	$1023, %rsi
	jg	.LBB0_2076
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2070
	.p2align	4, 0x90
.LBB0_2074:
	incq	%r8
	addq	$56, %rdi
.LBB0_2070:
	cmpq	$13, %r8
	jg	.LBB0_2075
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2074
	.p2align	4, 0x90
.LBB0_2073:
	movq	672(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2073
	jmp	.LBB0_2074
.LBB0_2077:
	xorl	%eax, %eax
	jmp	.LBB0_2078
	.p2align	4, 0x90
.LBB0_2097:
	incq	%rax
	addq	$200704, %rbx
.LBB0_2078:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2098
	movq	712(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_2080
	.p2align	4, 0x90
.LBB0_2096:
	incq	%rdx
	addq	$1024, -64(%rbp)
.LBB0_2080:
	cmpq	$1023, %rdx
	jg	.LBB0_2097
	movq	%rbx, %rcx
	xorl	%r11d, %r11d
	jmp	.LBB0_2082
	.p2align	4, 0x90
.LBB0_2095:
	incq	%r11
	movq	-72(%rbp), %rcx
	addq	$56, %rcx
.LBB0_2082:
	cmpq	$13, %r11
	jg	.LBB0_2096
	movq	%rcx, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_2084
	.p2align	4, 0x90
.LBB0_2094:
	incq	%r9
	movq	-56(%rbp), %rcx
	addq	$4, %rcx
.LBB0_2084:
	cmpq	$13, %r9
	jg	.LBB0_2095
	movq	%rcx, -56(%rbp)
	movq	%rcx, %r12
	movq	-64(%rbp), %r13
	xorl	%r14d, %r14d
	jmp	.LBB0_2086
	.p2align	4, 0x90
.LBB0_2093:
	incq	%r14
	movq	-80(%rbp), %r13
	addq	$4, %r13
	movq	-88(%rbp), %r12
	addq	$784, %r12
.LBB0_2086:
	cmpq	$255, %r14
	jg	.LBB0_2094
	movq	%r12, -88(%rbp)
	movq	%r13, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2088
	.p2align	4, 0x90
.LBB0_2092:
	incq	%rsi
	addq	$4, %r13
	addq	$56, %r12
.LBB0_2088:
	testq	%rsi, %rsi
	jg	.LBB0_2093
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_2092
	.p2align	4, 0x90
.LBB0_2091:
	movss	(%r12,%rcx,4), %xmm0
	imulq	$200704, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%r11, %rdi
	shlq	$4, %rdi
	subq	%r11, %rdi
	subq	%r11, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r15,%rdi,4), %xmm0
	movss	%xmm0, (%r15,%rdi,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_2091
	jmp	.LBB0_2092
.LBB0_2098:
	xorl	%eax, %eax
	movq	%r15, %rcx
	movq	-120(%rbp), %rbx
	jmp	.LBB0_2099
	.p2align	4, 0x90
.LBB0_2109:
	incq	%rax
	addq	$802816, %rbx
	addq	$802816, %rcx
.LBB0_2099:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2110
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2101
	.p2align	4, 0x90
.LBB0_2108:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_2101:
	cmpq	$1023, %rdi
	jg	.LBB0_2109
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2103
	.p2align	4, 0x90
.LBB0_2107:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_2103:
	cmpq	$13, %r10
	jg	.LBB0_2108
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_2107
	.p2align	4, 0x90
.LBB0_2106:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_2106
	jmp	.LBB0_2107
.LBB0_2110:
	movq	-128(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_2111
	.p2align	4, 0x90
.LBB0_2121:
	incq	%rbx
	addq	$802816, %rcx
	addq	$802816, %r15
.LBB0_2111:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_2122
	movq	%r15, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2113
	.p2align	4, 0x90
.LBB0_2120:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_2113:
	cmpq	$1023, %rdi
	jg	.LBB0_2121
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2115
	.p2align	4, 0x90
.LBB0_2119:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_2115:
	cmpq	$13, %r10
	jg	.LBB0_2120
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_2119
	.p2align	4, 0x90
.LBB0_2118:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_2118
	jmp	.LBB0_2119
.LBB0_2122:
	movq	%rax, -120(%rbp)
	movq	-112(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -112(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	544(%rbp), %r10
	jmp	.LBB0_2123
	.p2align	4, 0x90
.LBB0_2133:
	incq	%rax
	addq	$200704, %rcx
.LBB0_2123:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2134
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2125
	.p2align	4, 0x90
.LBB0_2132:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2125:
	cmpq	$255, %rsi
	jg	.LBB0_2133
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2127
	.p2align	4, 0x90
.LBB0_2131:
	incq	%r8
	addq	$56, %rdi
.LBB0_2127:
	cmpq	$13, %r8
	jg	.LBB0_2132
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2131
	.p2align	4, 0x90
.LBB0_2130:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2130
	jmp	.LBB0_2131
.LBB0_2134:
	xorl	%eax, %eax
	movq	-120(%rbp), %rcx
	movq	%rcx, -96(%rbp)
	jmp	.LBB0_2135
	.p2align	4, 0x90
.LBB0_2154:
	incq	%rax
	addq	$802816, -96(%rbp)
.LBB0_2135:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2155
	movq	584(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2137
	.p2align	4, 0x90
.LBB0_2153:
	incq	%rsi
	addq	$4096, -64(%rbp)
.LBB0_2137:
	cmpq	$255, %rsi
	jg	.LBB0_2154
	movq	-96(%rbp), %r11
	xorl	%edx, %edx
	jmp	.LBB0_2139
	.p2align	4, 0x90
.LBB0_2152:
	incq	%rdx
	movq	-72(%rbp), %r11
	addq	$56, %r11
.LBB0_2139:
	cmpq	$13, %rdx
	jg	.LBB0_2153
	movq	%r11, -72(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_2141
	.p2align	4, 0x90
.LBB0_2151:
	incq	%r10
	movq	-56(%rbp), %r11
	addq	$4, %r11
.LBB0_2141:
	cmpq	$13, %r10
	jg	.LBB0_2152
	movq	%r11, -56(%rbp)
	movq	-64(%rbp), %r14
	xorl	%r15d, %r15d
	jmp	.LBB0_2143
	.p2align	4, 0x90
.LBB0_2150:
	incq	%r15
	movq	-80(%rbp), %r14
	addq	$4, %r14
	movq	-88(%rbp), %r11
	addq	$784, %r11
.LBB0_2143:
	cmpq	$1023, %r15
	jg	.LBB0_2151
	movq	%r11, -88(%rbp)
	movq	%r14, -80(%rbp)
	xorl	%ecx, %ecx
	jmp	.LBB0_2145
	.p2align	4, 0x90
.LBB0_2149:
	incq	%rcx
	addq	$4, %r14
	addq	$56, %r11
.LBB0_2145:
	testq	%rcx, %rcx
	jg	.LBB0_2150
	xorl	%r13d, %r13d
	testq	%r13, %r13
	jg	.LBB0_2149
	.p2align	4, 0x90
.LBB0_2148:
	movss	(%r11,%r13,4), %xmm0
	imulq	$50176, %rax, %rdi
	imulq	$196, %rsi, %r9
	addq	%rdi, %r9
	movq	%rdx, %r8
	shlq	$4, %r8
	subq	%rdx, %r8
	subq	%rdx, %r8
	addq	%r9, %r8
	addq	%r10, %r8
	mulss	(%r14,%r13,4), %xmm0
	addss	(%rbx,%r8,4), %xmm0
	movss	%xmm0, (%rbx,%r8,4)
	incq	%r13
	testq	%r13, %r13
	jle	.LBB0_2148
	jmp	.LBB0_2149
.LBB0_2155:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_2156
	.p2align	4, 0x90
.LBB0_2166:
	incq	%rax
	addq	$200704, %rcx
.LBB0_2156:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2167
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2158
	.p2align	4, 0x90
.LBB0_2165:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2158:
	cmpq	$255, %rsi
	jg	.LBB0_2166
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2160
	.p2align	4, 0x90
.LBB0_2164:
	incq	%r8
	addq	$56, %rdi
.LBB0_2160:
	cmpq	$13, %r8
	jg	.LBB0_2165
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2164
	.p2align	4, 0x90
.LBB0_2163:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2163
	jmp	.LBB0_2164
.LBB0_2167:
	movq	-104(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -96(%rbp)
	jmp	.LBB0_2168
	.p2align	4, 0x90
.LBB0_2178:
	incq	%rcx
	addq	$262144, %rdx
.LBB0_2168:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2179
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2170
	.p2align	4, 0x90
.LBB0_2177:
	incq	%rdi
	addq	$1024, %rsi
.LBB0_2170:
	cmpq	$255, %rdi
	jg	.LBB0_2178
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_2172
	.p2align	4, 0x90
.LBB0_2176:
	incq	%r9
	addq	$64, %r8
.LBB0_2172:
	cmpq	$15, %r9
	jg	.LBB0_2177
	xorl	%r10d, %r10d
	cmpq	$15, %r10
	jg	.LBB0_2176
	.p2align	4, 0x90
.LBB0_2175:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$15, %r10
	jle	.LBB0_2175
	jmp	.LBB0_2176
.LBB0_2179:
	movq	%rsp, %r15
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$14, %r8d
	movq	%r8, -56(%rdx)
	movl	$256, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-112(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$50176, %edi
	movq	%rdi, -40(%rdx)
	movl	$196, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-96(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$16, -24(%rdx)
	movq	$256, -32(%rdx)
	movq	$65536, -40(%rdx)
	movq	$14, -48(%rdx)
	movq	$14, -56(%rdx)
	movq	$256, -64(%rdx)
	movq	$17, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r15, %rsp
	movq	%rbx, %rcx
	movq	416(%rbp), %r10
	jmp	.LBB0_2180
	.p2align	4, 0x90
.LBB0_2190:
	incq	%rax
	addq	$200704, %rcx
.LBB0_2180:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2191
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2182
	.p2align	4, 0x90
.LBB0_2189:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2182:
	cmpq	$255, %rsi
	jg	.LBB0_2190
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2184
	.p2align	4, 0x90
.LBB0_2188:
	incq	%r8
	addq	$56, %rdi
.LBB0_2184:
	cmpq	$13, %r8
	jg	.LBB0_2189
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2188
	.p2align	4, 0x90
.LBB0_2187:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2187
	jmp	.LBB0_2188
.LBB0_2191:
	xorl	%eax, %eax
	jmp	.LBB0_2192
	.p2align	4, 0x90
.LBB0_2211:
	incq	%rax
	addq	$262144, -96(%rbp)
.LBB0_2192:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2212
	movq	456(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_2194
	.p2align	4, 0x90
.LBB0_2210:
	incq	%rdx
	addq	$9216, -64(%rbp)
.LBB0_2194:
	cmpq	$255, %rdx
	jg	.LBB0_2211
	movq	-96(%rbp), %rsi
	xorl	%ecx, %ecx
	jmp	.LBB0_2196
	.p2align	4, 0x90
.LBB0_2209:
	incq	%rcx
	movq	-72(%rbp), %rsi
	addq	$64, %rsi
.LBB0_2196:
	cmpq	$13, %rcx
	jg	.LBB0_2210
	movq	%rsi, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_2198
	.p2align	4, 0x90
.LBB0_2208:
	incq	%r9
	movq	-56(%rbp), %rsi
	addq	$4, %rsi
.LBB0_2198:
	cmpq	$13, %r9
	jg	.LBB0_2209
	movq	%rsi, -56(%rbp)
	movq	%rsi, %r13
	movq	-64(%rbp), %r11
	xorl	%r15d, %r15d
	jmp	.LBB0_2200
	.p2align	4, 0x90
.LBB0_2207:
	incq	%r15
	movq	-80(%rbp), %r11
	addq	$36, %r11
	movq	-88(%rbp), %r13
	addq	$1024, %r13
.LBB0_2200:
	cmpq	$255, %r15
	jg	.LBB0_2208
	movq	%r13, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2202
	.p2align	4, 0x90
.LBB0_2206:
	incq	%rsi
	addq	$12, %r11
	addq	$64, %r13
.LBB0_2202:
	cmpq	$2, %rsi
	jg	.LBB0_2207
	xorl	%r14d, %r14d
	cmpq	$2, %r14
	jg	.LBB0_2206
	.p2align	4, 0x90
.LBB0_2205:
	movss	(%r13,%r14,4), %xmm0
	imulq	$50176, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%rcx, %rdi
	shlq	$4, %rdi
	subq	%rcx, %rdi
	subq	%rcx, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r11,%r14,4), %xmm0
	addss	(%rbx,%rdi,4), %xmm0
	movss	%xmm0, (%rbx,%rdi,4)
	incq	%r14
	cmpq	$2, %r14
	jle	.LBB0_2205
	jmp	.LBB0_2206
.LBB0_2212:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_2213
	.p2align	4, 0x90
.LBB0_2223:
	incq	%rax
	addq	$200704, %rcx
.LBB0_2213:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2224
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2215
	.p2align	4, 0x90
.LBB0_2222:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2215:
	cmpq	$255, %rsi
	jg	.LBB0_2223
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2217
	.p2align	4, 0x90
.LBB0_2221:
	incq	%r8
	addq	$56, %rdi
.LBB0_2217:
	cmpq	$13, %r8
	jg	.LBB0_2222
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2221
	.p2align	4, 0x90
.LBB0_2220:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2220
	jmp	.LBB0_2221
.LBB0_2224:
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_2225
	.p2align	4, 0x90
.LBB0_2235:
	incq	%rax
	addq	$802816, %rcx
.LBB0_2225:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2236
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2227
	.p2align	4, 0x90
.LBB0_2234:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2227:
	cmpq	$1023, %rsi
	jg	.LBB0_2235
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2229
	.p2align	4, 0x90
.LBB0_2233:
	incq	%r8
	addq	$56, %rdi
.LBB0_2229:
	cmpq	$13, %r8
	jg	.LBB0_2234
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2233
	.p2align	4, 0x90
.LBB0_2232:
	movq	288(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2232
	jmp	.LBB0_2233
.LBB0_2236:
	xorl	%eax, %eax
	jmp	.LBB0_2237
	.p2align	4, 0x90
.LBB0_2256:
	incq	%rax
	addq	$200704, %rbx
.LBB0_2237:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2257
	movq	328(%rbp), %rcx
	movq	%rcx, -64(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_2239
	.p2align	4, 0x90
.LBB0_2255:
	incq	%rdx
	addq	$1024, -64(%rbp)
.LBB0_2239:
	cmpq	$1023, %rdx
	jg	.LBB0_2256
	movq	%rbx, %rcx
	xorl	%r11d, %r11d
	jmp	.LBB0_2241
	.p2align	4, 0x90
.LBB0_2254:
	incq	%r11
	movq	-72(%rbp), %rcx
	addq	$56, %rcx
.LBB0_2241:
	cmpq	$13, %r11
	jg	.LBB0_2255
	movq	%rcx, -72(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_2243
	.p2align	4, 0x90
.LBB0_2253:
	incq	%r9
	movq	-56(%rbp), %rcx
	addq	$4, %rcx
.LBB0_2243:
	cmpq	$13, %r9
	jg	.LBB0_2254
	movq	%rcx, -56(%rbp)
	movq	%rcx, %r15
	movq	-64(%rbp), %r13
	xorl	%r14d, %r14d
	jmp	.LBB0_2245
	.p2align	4, 0x90
.LBB0_2252:
	incq	%r14
	movq	-80(%rbp), %r13
	addq	$4, %r13
	movq	-88(%rbp), %r15
	addq	$784, %r15
.LBB0_2245:
	cmpq	$255, %r14
	jg	.LBB0_2253
	movq	%r15, -88(%rbp)
	movq	%r13, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2247
	.p2align	4, 0x90
.LBB0_2251:
	incq	%rsi
	addq	$4, %r13
	addq	$56, %r15
.LBB0_2247:
	testq	%rsi, %rsi
	jg	.LBB0_2252
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_2251
	.p2align	4, 0x90
.LBB0_2250:
	movss	(%r15,%rcx,4), %xmm0
	imulq	$200704, %rax, %r8
	imulq	$196, %rdx, %r10
	addq	%r8, %r10
	movq	%r11, %rdi
	shlq	$4, %rdi
	subq	%r11, %rdi
	subq	%r11, %rdi
	addq	%r10, %rdi
	addq	%r9, %rdi
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r12,%rdi,4), %xmm0
	movss	%xmm0, (%r12,%rdi,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_2250
	jmp	.LBB0_2251
.LBB0_2257:
	xorl	%eax, %eax
	movq	%r12, %rcx
	movq	-120(%rbp), %rbx
	jmp	.LBB0_2258
	.p2align	4, 0x90
.LBB0_2268:
	incq	%rax
	addq	$802816, %rbx
	addq	$802816, %rcx
.LBB0_2258:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2269
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2260
	.p2align	4, 0x90
.LBB0_2267:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_2260:
	cmpq	$1023, %rdi
	jg	.LBB0_2268
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2262
	.p2align	4, 0x90
.LBB0_2266:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_2262:
	cmpq	$13, %r10
	jg	.LBB0_2267
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_2266
	.p2align	4, 0x90
.LBB0_2265:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_2265
	jmp	.LBB0_2266
.LBB0_2269:
	movq	-128(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rcx
	addq	$63, %rcx
	andq	$-64, %rcx
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rcx, -104(%rbp)
	jmp	.LBB0_2270
	.p2align	4, 0x90
.LBB0_2280:
	incq	%rax
	addq	$802816, %rcx
	addq	$802816, %r12
.LBB0_2270:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2281
	movq	%r12, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2272
	.p2align	4, 0x90
.LBB0_2279:
	incq	%rdi
	addq	$784, %rsi
	addq	$784, %rdx
.LBB0_2272:
	cmpq	$1023, %rdi
	jg	.LBB0_2280
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2274
	.p2align	4, 0x90
.LBB0_2278:
	incq	%r10
	addq	$56, %r9
	addq	$56, %r8
.LBB0_2274:
	cmpq	$13, %r10
	jg	.LBB0_2279
	xorl	%r11d, %r11d
	cmpq	$13, %r11
	jg	.LBB0_2278
	.p2align	4, 0x90
.LBB0_2277:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$13, %r11
	jle	.LBB0_2277
	jmp	.LBB0_2278
.LBB0_2281:
	movq	-136(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r13
	addq	$63, %r13
	andq	$-64, %r13
	xorl	%eax, %eax
	movq	%r13, %rcx
	movq	160(%rbp), %r10
	jmp	.LBB0_2282
	.p2align	4, 0x90
.LBB0_2292:
	incq	%rax
	addq	$401408, %rcx
.LBB0_2282:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2293
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2284
	.p2align	4, 0x90
.LBB0_2291:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2284:
	cmpq	$2047, %rsi
	jg	.LBB0_2292
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2286
	.p2align	4, 0x90
.LBB0_2290:
	incq	%r8
	addq	$28, %rdi
.LBB0_2286:
	cmpq	$6, %r8
	jg	.LBB0_2291
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2290
	.p2align	4, 0x90
.LBB0_2289:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2289
	jmp	.LBB0_2290
.LBB0_2293:
	xorl	%eax, %eax
	movq	-104(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	jmp	.LBB0_2294
	.p2align	4, 0x90
.LBB0_2313:
	incq	%rax
	addq	$802816, -72(%rbp)
.LBB0_2294:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2314
	movq	200(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2296
	.p2align	4, 0x90
.LBB0_2312:
	incq	%rsi
	addq	$4096, -56(%rbp)
.LBB0_2296:
	cmpq	$2047, %rsi
	jg	.LBB0_2313
	movq	-72(%rbp), %rcx
	xorl	%r8d, %r8d
	jmp	.LBB0_2298
	.p2align	4, 0x90
.LBB0_2311:
	incq	%r8
	movq	-64(%rbp), %rcx
	addq	$112, %rcx
.LBB0_2298:
	cmpq	$6, %r8
	jg	.LBB0_2312
	movq	%rcx, -64(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_2300
	.p2align	4, 0x90
.LBB0_2310:
	incq	%r10
	movq	-80(%rbp), %rcx
	addq	$8, %rcx
.LBB0_2300:
	cmpq	$6, %r10
	jg	.LBB0_2311
	movq	%rcx, -80(%rbp)
	movq	%rcx, %r12
	movq	-56(%rbp), %rdx
	xorl	%r14d, %r14d
	jmp	.LBB0_2302
	.p2align	4, 0x90
.LBB0_2309:
	incq	%r14
	addq	$4, %rdx
	movq	-88(%rbp), %r12
	addq	$784, %r12
.LBB0_2302:
	cmpq	$1023, %r14
	jg	.LBB0_2310
	movq	%r12, -88(%rbp)
	movq	%rdx, %rbx
	xorl	%edi, %edi
	jmp	.LBB0_2304
	.p2align	4, 0x90
.LBB0_2308:
	incq	%rdi
	addq	$4, %rbx
	addq	$56, %r12
.LBB0_2304:
	testq	%rdi, %rdi
	jg	.LBB0_2309
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_2308
	.p2align	4, 0x90
.LBB0_2307:
	movss	(%r12,%rcx,4), %xmm0
	imulq	$100352, %rax, %r9
	imulq	$49, %rsi, %r11
	addq	%r9, %r11
	leaq	(,%r8,8), %r9
	subq	%r8, %r9
	addq	%r11, %r9
	addq	%r10, %r9
	mulss	(%rbx,%rcx,4), %xmm0
	addss	(%r13,%r9,4), %xmm0
	movss	%xmm0, (%r13,%r9,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_2307
	jmp	.LBB0_2308
.LBB0_2314:
	movq	-136(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -128(%rbp)
	leaq	63(%rax), %r14
	andq	$-64, %r14
	xorl	%eax, %eax
	movq	%r14, %rcx
	movq	7072(%rbp), %r10
	jmp	.LBB0_2315
	.p2align	4, 0x90
.LBB0_2325:
	incq	%rax
	addq	$401408, %rcx
.LBB0_2315:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2326
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2317
	.p2align	4, 0x90
.LBB0_2324:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2317:
	cmpq	$511, %rsi
	jg	.LBB0_2325
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2319
	.p2align	4, 0x90
.LBB0_2323:
	incq	%r8
	addq	$56, %rdi
.LBB0_2319:
	cmpq	$13, %r8
	jg	.LBB0_2324
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2323
	.p2align	4, 0x90
.LBB0_2322:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2322
	jmp	.LBB0_2323
.LBB0_2326:
	xorl	%eax, %eax
	jmp	.LBB0_2327
	.p2align	4, 0x90
.LBB0_2346:
	incq	%rax
	addq	$802816, -104(%rbp)
.LBB0_2327:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2347
	movq	72(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_2329
	.p2align	4, 0x90
.LBB0_2345:
	incq	%rdx
	addq	$4096, -72(%rbp)
.LBB0_2329:
	cmpq	$511, %rdx
	jg	.LBB0_2346
	movq	-104(%rbp), %rcx
	xorl	%ebx, %ebx
	jmp	.LBB0_2331
	.p2align	4, 0x90
.LBB0_2344:
	incq	%rbx
	movq	-96(%rbp), %rcx
	addq	$56, %rcx
.LBB0_2331:
	cmpq	$13, %rbx
	jg	.LBB0_2345
	movq	%rcx, -96(%rbp)
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_2333
	.p2align	4, 0x90
.LBB0_2343:
	incq	%r9
	addq	$4, -64(%rbp)
.LBB0_2333:
	cmpq	$13, %r9
	jg	.LBB0_2344
	movq	-64(%rbp), %r12
	movq	-72(%rbp), %r11
	xorl	%ecx, %ecx
	jmp	.LBB0_2335
	.p2align	4, 0x90
.LBB0_2342:
	movq	-56(%rbp), %rcx
	incq	%rcx
	movq	-80(%rbp), %r11
	addq	$4, %r11
	movq	-88(%rbp), %r12
	addq	$784, %r12
.LBB0_2335:
	cmpq	$1023, %rcx
	jg	.LBB0_2343
	movq	%rcx, -56(%rbp)
	movq	%r12, -88(%rbp)
	movq	%r11, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2337
	.p2align	4, 0x90
.LBB0_2341:
	incq	%rsi
	addq	$4, %r11
	addq	$56, %r12
.LBB0_2337:
	testq	%rsi, %rsi
	jg	.LBB0_2342
	xorl	%r8d, %r8d
	testq	%r8, %r8
	jg	.LBB0_2341
	.p2align	4, 0x90
.LBB0_2340:
	movss	(%r12,%r8,4), %xmm0
	imulq	$100352, %rax, %r10
	imulq	$196, %rdx, %rcx
	addq	%r10, %rcx
	movq	%rbx, %rdi
	shlq	$4, %rdi
	subq	%rbx, %rdi
	subq	%rbx, %rdi
	addq	%rcx, %rdi
	addq	%r9, %rdi
	mulss	(%r11,%r8,4), %xmm0
	addss	(%r14,%rdi,4), %xmm0
	movss	%xmm0, (%r14,%rdi,4)
	incq	%r8
	testq	%r8, %r8
	jle	.LBB0_2340
	jmp	.LBB0_2341
.LBB0_2347:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r14, %rcx
	jmp	.LBB0_2348
	.p2align	4, 0x90
.LBB0_2358:
	incq	%rax
	addq	$401408, %rcx
.LBB0_2348:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2359
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2350
	.p2align	4, 0x90
.LBB0_2357:
	incq	%rsi
	addq	$784, %rdx
.LBB0_2350:
	cmpq	$511, %rsi
	jg	.LBB0_2358
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2352
	.p2align	4, 0x90
.LBB0_2356:
	incq	%r8
	addq	$56, %rdi
.LBB0_2352:
	cmpq	$13, %r8
	jg	.LBB0_2357
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_2356
	.p2align	4, 0x90
.LBB0_2355:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_2355
	jmp	.LBB0_2356
.LBB0_2359:
	movq	-48(%rbp), %rdi
	shlq	$19, %rdi
	orq	$64, %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -96(%rbp)
	jmp	.LBB0_2360
	.p2align	4, 0x90
.LBB0_2370:
	incq	%rcx
	addq	$524288, %rdx
.LBB0_2360:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2371
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2362
	.p2align	4, 0x90
.LBB0_2369:
	incq	%rdi
	addq	$1024, %rsi
.LBB0_2362:
	cmpq	$511, %rdi
	jg	.LBB0_2370
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_2364
	.p2align	4, 0x90
.LBB0_2368:
	incq	%r9
	addq	$64, %r8
.LBB0_2364:
	cmpq	$15, %r9
	jg	.LBB0_2369
	xorl	%r10d, %r10d
	cmpq	$15, %r10
	jg	.LBB0_2368
	.p2align	4, 0x90
.LBB0_2367:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$15, %r10
	jle	.LBB0_2367
	jmp	.LBB0_2368
.LBB0_2371:
	movq	%rsp, %r12
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$14, %edi
	movq	%rdi, -56(%rdx)
	movl	$512, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rbx
	movq	%rbx, -72(%rdx)
	movq	%r14, -88(%rdx)
	movq	-128(%rbp), %rsi
	movq	%rsi, -96(%rdx)
	movq	%rdi, -48(%rdx)
	movl	$100352, %esi
	movq	%rsi, -40(%rdx)
	movl	$196, %esi
	movq	%rsi, -32(%rdx)
	movq	%rdi, -24(%rdx)
	movl	$1, %esi
	movq	%rsi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rbx, -72(%rdx)
	movq	-96(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$16, -24(%rdx)
	movq	$256, -32(%rdx)
	movq	$131072, -40(%rdx)
	movq	$14, -48(%rdx)
	movq	$14, -56(%rdx)
	movq	$512, -64(%rdx)
	movq	$17, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	movq	%r12, %rsp
	imulq	$100352, %rbx, %rdi
	orq	$64, %rdi
	movq	%rdi, -104(%rbp)
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ecx, %ecx
	movq	%rax, %rdx
	movq	6944(%rbp), %r11
	jmp	.LBB0_2372
	.p2align	4, 0x90
.LBB0_2382:
	incq	%rcx
	addq	$100352, %rdx
.LBB0_2372:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2383
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2374
	.p2align	4, 0x90
.LBB0_2381:
	incq	%rdi
	addq	$196, %rsi
.LBB0_2374:
	cmpq	$511, %rdi
	jg	.LBB0_2382
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_2376
	.p2align	4, 0x90
.LBB0_2380:
	incq	%r9
	addq	$28, %r8
.LBB0_2376:
	cmpq	$6, %r9
	jg	.LBB0_2381
	xorl	%r10d, %r10d
	cmpq	$6, %r10
	jg	.LBB0_2380
	.p2align	4, 0x90
.LBB0_2379:
	movss	(%r11,%rdi,4), %xmm0
	movss	%xmm0, (%r8,%r10,4)
	incq	%r10
	cmpq	$6, %r10
	jle	.LBB0_2379
	jmp	.LBB0_2380
.LBB0_2383:
	xorl	%ecx, %ecx
	jmp	.LBB0_2384
	.p2align	4, 0x90
.LBB0_2403:
	incq	%rcx
	addq	$524288, -96(%rbp)
.LBB0_2384:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2404
	movq	6984(%rbp), %rdx
	movq	%rdx, -64(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2386
	.p2align	4, 0x90
.LBB0_2402:
	incq	%rsi
	addq	$18432, -64(%rbp)
.LBB0_2386:
	cmpq	$511, %rsi
	jg	.LBB0_2403
	movq	-96(%rbp), %r11
	xorl	%r8d, %r8d
	jmp	.LBB0_2388
	.p2align	4, 0x90
.LBB0_2401:
	incq	%r8
	movq	-72(%rbp), %r11
	subq	$-128, %r11
.LBB0_2388:
	cmpq	$6, %r8
	jg	.LBB0_2402
	movq	%r11, -72(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_2390
	.p2align	4, 0x90
.LBB0_2400:
	incq	%r10
	movq	-56(%rbp), %r11
	addq	$8, %r11
.LBB0_2390:
	cmpq	$6, %r10
	jg	.LBB0_2401
	movq	%r11, -56(%rbp)
	movq	-64(%rbp), %r14
	xorl	%r12d, %r12d
	jmp	.LBB0_2392
	.p2align	4, 0x90
.LBB0_2399:
	incq	%r12
	movq	-80(%rbp), %r14
	addq	$36, %r14
	movq	-88(%rbp), %r11
	addq	$1024, %r11
.LBB0_2392:
	cmpq	$511, %r12
	jg	.LBB0_2400
	movq	%r11, -88(%rbp)
	movq	%r14, -80(%rbp)
	xorl	%ebx, %ebx
	jmp	.LBB0_2394
	.p2align	4, 0x90
.LBB0_2398:
	incq	%rbx
	addq	$12, %r14
	addq	$64, %r11
.LBB0_2394:
	cmpq	$2, %rbx
	jg	.LBB0_2399
	xorl	%edi, %edi
	cmpq	$2, %rdi
	jg	.LBB0_2398
	.p2align	4, 0x90
.LBB0_2397:
	movss	(%r11,%rdi,4), %xmm0
	imulq	$25088, %rcx, %r9
	imulq	$49, %rsi, %rdx
	addq	%r9, %rdx
	leaq	(,%r8,8), %r9
	subq	%r8, %r9
	addq	%rdx, %r9
	addq	%r10, %r9
	mulss	(%r14,%rdi,4), %xmm0
	addss	(%rax,%r9,4), %xmm0
	movss	%xmm0, (%rax,%r9,4)
	incq	%rdi
	cmpq	$2, %rdi
	jle	.LBB0_2397
	jmp	.LBB0_2398
.LBB0_2404:
	xorl	%ecx, %ecx
	xorps	%xmm0, %xmm0
	movq	%rax, %rdx
	jmp	.LBB0_2405
	.p2align	4, 0x90
.LBB0_2415:
	incq	%rcx
	addq	$100352, %rdx
.LBB0_2405:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2416
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2407
	.p2align	4, 0x90
.LBB0_2414:
	incq	%rdi
	addq	$196, %rsi
.LBB0_2407:
	cmpq	$511, %rdi
	jg	.LBB0_2415
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_2409
	.p2align	4, 0x90
.LBB0_2413:
	incq	%r9
	addq	$28, %r8
.LBB0_2409:
	cmpq	$6, %r9
	jg	.LBB0_2414
	xorl	%r10d, %r10d
	cmpq	$6, %r10
	jg	.LBB0_2413
	.p2align	4, 0x90
.LBB0_2412:
	movss	(%r8,%r10,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r8,%r10,4)
	incq	%r10
	cmpq	$6, %r10
	jle	.LBB0_2412
	jmp	.LBB0_2413
.LBB0_2416:
	xorl	%ecx, %ecx
	movq	%r15, %rdx
	jmp	.LBB0_2417
	.p2align	4, 0x90
.LBB0_2427:
	incq	%rcx
	addq	$401408, %rdx
.LBB0_2417:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2428
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2419
	.p2align	4, 0x90
.LBB0_2426:
	incq	%rdi
	addq	$196, %rsi
.LBB0_2419:
	cmpq	$2047, %rdi
	jg	.LBB0_2427
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_2421
	.p2align	4, 0x90
.LBB0_2425:
	incq	%r9
	addq	$28, %r8
.LBB0_2421:
	cmpq	$6, %r9
	jg	.LBB0_2426
	xorl	%r10d, %r10d
	cmpq	$6, %r10
	jg	.LBB0_2425
	.p2align	4, 0x90
.LBB0_2424:
	movq	6728(%rbp), %r11
	movss	(%r11,%rdi,4), %xmm0
	movss	%xmm0, (%r8,%r10,4)
	incq	%r10
	cmpq	$6, %r10
	jle	.LBB0_2424
	jmp	.LBB0_2425
.LBB0_2428:
	xorl	%ecx, %ecx
	jmp	.LBB0_2429
	.p2align	4, 0x90
.LBB0_2448:
	incq	%rcx
	addq	$100352, %rax
.LBB0_2429:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2449
	movq	6856(%rbp), %rdx
	movq	%rdx, -64(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2431
	.p2align	4, 0x90
.LBB0_2447:
	incq	%rsi
	addq	$2048, -64(%rbp)
.LBB0_2431:
	cmpq	$2047, %rsi
	jg	.LBB0_2448
	movq	%rax, %rdx
	xorl	%r8d, %r8d
	jmp	.LBB0_2433
	.p2align	4, 0x90
.LBB0_2446:
	incq	%r8
	movq	-72(%rbp), %rdx
	addq	$28, %rdx
.LBB0_2433:
	cmpq	$6, %r8
	jg	.LBB0_2447
	movq	%rdx, -72(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_2435
	.p2align	4, 0x90
.LBB0_2445:
	incq	%r10
	movq	-56(%rbp), %rdx
	addq	$4, %rdx
.LBB0_2435:
	cmpq	$6, %r10
	jg	.LBB0_2446
	movq	%rdx, -56(%rbp)
	movq	%rdx, %r12
	movq	-64(%rbp), %rbx
	xorl	%r14d, %r14d
	jmp	.LBB0_2437
	.p2align	4, 0x90
.LBB0_2444:
	incq	%r14
	movq	-80(%rbp), %rbx
	addq	$4, %rbx
	movq	-88(%rbp), %r12
	addq	$196, %r12
.LBB0_2437:
	cmpq	$511, %r14
	jg	.LBB0_2445
	movq	%r12, -88(%rbp)
	movq	%rbx, -80(%rbp)
	xorl	%edi, %edi
	jmp	.LBB0_2439
	.p2align	4, 0x90
.LBB0_2443:
	incq	%rdi
	addq	$4, %rbx
	addq	$28, %r12
.LBB0_2439:
	testq	%rdi, %rdi
	jg	.LBB0_2444
	xorl	%r9d, %r9d
	testq	%r9, %r9
	jg	.LBB0_2443
	.p2align	4, 0x90
.LBB0_2442:
	movss	(%r12,%r9,4), %xmm0
	imulq	$100352, %rcx, %r11
	imulq	$49, %rsi, %rdx
	addq	%r11, %rdx
	leaq	(,%r8,8), %r11
	subq	%r8, %r11
	addq	%rdx, %r11
	addq	%r10, %r11
	mulss	(%rbx,%r9,4), %xmm0
	addss	(%r15,%r11,4), %xmm0
	movss	%xmm0, (%r15,%r11,4)
	incq	%r9
	testq	%r9, %r9
	jle	.LBB0_2442
	jmp	.LBB0_2443
.LBB0_2449:
	xorl	%eax, %eax
	movq	%r15, %rcx
	jmp	.LBB0_2450
	.p2align	4, 0x90
.LBB0_2460:
	incq	%rax
	addq	$401408, %r13
	addq	$401408, %rcx
.LBB0_2450:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2461
	movq	%rcx, %rdx
	movq	%r13, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2452
	.p2align	4, 0x90
.LBB0_2459:
	incq	%rdi
	addq	$196, %rsi
	addq	$196, %rdx
.LBB0_2452:
	cmpq	$2047, %rdi
	jg	.LBB0_2460
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2454
	.p2align	4, 0x90
.LBB0_2458:
	incq	%r10
	addq	$28, %r9
	addq	$28, %r8
.LBB0_2454:
	cmpq	$6, %r10
	jg	.LBB0_2459
	xorl	%r11d, %r11d
	cmpq	$6, %r11
	jg	.LBB0_2458
	.p2align	4, 0x90
.LBB0_2457:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$6, %r11
	jle	.LBB0_2457
	jmp	.LBB0_2458
.LBB0_2461:
	movq	-136(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_2462
	.p2align	4, 0x90
.LBB0_2472:
	incq	%rbx
	addq	$401408, %rcx
	addq	$401408, %r15
.LBB0_2462:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_2473
	movq	%r15, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2464
	.p2align	4, 0x90
.LBB0_2471:
	incq	%rdi
	addq	$196, %rsi
	addq	$196, %rdx
.LBB0_2464:
	cmpq	$2047, %rdi
	jg	.LBB0_2472
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2466
	.p2align	4, 0x90
.LBB0_2470:
	incq	%r10
	addq	$28, %r9
	addq	$28, %r8
.LBB0_2466:
	cmpq	$6, %r10
	jg	.LBB0_2471
	xorl	%r11d, %r11d
	cmpq	$6, %r11
	jg	.LBB0_2470
	.p2align	4, 0x90
.LBB0_2469:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$6, %r11
	jle	.LBB0_2469
	jmp	.LBB0_2470
.LBB0_2473:
	movq	%rax, -96(%rbp)
	movq	-104(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -112(%rbp)
	leaq	63(%rax), %r14
	andq	$-64, %r14
	xorl	%eax, %eax
	movq	%r14, %rcx
	movq	6472(%rbp), %r10
	jmp	.LBB0_2474
	.p2align	4, 0x90
.LBB0_2484:
	incq	%rax
	addq	$100352, %rcx
.LBB0_2474:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2485
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2476
	.p2align	4, 0x90
.LBB0_2483:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2476:
	cmpq	$511, %rsi
	jg	.LBB0_2484
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2478
	.p2align	4, 0x90
.LBB0_2482:
	incq	%r8
	addq	$28, %rdi
.LBB0_2478:
	cmpq	$6, %r8
	jg	.LBB0_2483
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2482
	.p2align	4, 0x90
.LBB0_2481:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2481
	jmp	.LBB0_2482
.LBB0_2485:
	xorl	%eax, %eax
	movq	-96(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	jmp	.LBB0_2486
	.p2align	4, 0x90
.LBB0_2505:
	incq	%rax
	addq	$401408, -72(%rbp)
.LBB0_2486:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2506
	movq	6600(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2488
	.p2align	4, 0x90
.LBB0_2504:
	incq	%rsi
	addq	$8192, -56(%rbp)
.LBB0_2488:
	cmpq	$511, %rsi
	jg	.LBB0_2505
	movq	-72(%rbp), %r11
	xorl	%r8d, %r8d
	jmp	.LBB0_2490
	.p2align	4, 0x90
.LBB0_2503:
	incq	%r8
	movq	-64(%rbp), %r11
	addq	$28, %r11
.LBB0_2490:
	cmpq	$6, %r8
	jg	.LBB0_2504
	movq	%r11, -64(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_2492
	.p2align	4, 0x90
.LBB0_2502:
	incq	%r10
	movq	-80(%rbp), %r11
	addq	$4, %r11
.LBB0_2492:
	cmpq	$6, %r10
	jg	.LBB0_2503
	movq	%r11, -80(%rbp)
	movq	-56(%rbp), %rdx
	xorl	%r15d, %r15d
	jmp	.LBB0_2494
	.p2align	4, 0x90
.LBB0_2501:
	incq	%r15
	addq	$4, %rdx
	movq	-88(%rbp), %r11
	addq	$196, %r11
.LBB0_2494:
	cmpq	$2047, %r15
	jg	.LBB0_2502
	movq	%r11, -88(%rbp)
	movq	%rdx, %r13
	xorl	%ecx, %ecx
	jmp	.LBB0_2496
	.p2align	4, 0x90
.LBB0_2500:
	incq	%rcx
	addq	$4, %r13
	addq	$28, %r11
.LBB0_2496:
	testq	%rcx, %rcx
	jg	.LBB0_2501
	xorl	%ebx, %ebx
	testq	%rbx, %rbx
	jg	.LBB0_2500
	.p2align	4, 0x90
.LBB0_2499:
	movss	(%r11,%rbx,4), %xmm0
	imulq	$25088, %rax, %rdi
	imulq	$49, %rsi, %r9
	addq	%rdi, %r9
	leaq	(,%r8,8), %rdi
	subq	%r8, %rdi
	addq	%r9, %rdi
	addq	%r10, %rdi
	mulss	(%r13,%rbx,4), %xmm0
	addss	(%r14,%rdi,4), %xmm0
	movss	%xmm0, (%r14,%rdi,4)
	incq	%rbx
	testq	%rbx, %rbx
	jle	.LBB0_2499
	jmp	.LBB0_2500
.LBB0_2506:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r14, %rcx
	jmp	.LBB0_2507
	.p2align	4, 0x90
.LBB0_2517:
	incq	%rax
	addq	$100352, %rcx
.LBB0_2507:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2518
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2509
	.p2align	4, 0x90
.LBB0_2516:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2509:
	cmpq	$511, %rsi
	jg	.LBB0_2517
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2511
	.p2align	4, 0x90
.LBB0_2515:
	incq	%r8
	addq	$28, %rdi
.LBB0_2511:
	cmpq	$6, %r8
	jg	.LBB0_2516
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2515
	.p2align	4, 0x90
.LBB0_2514:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2514
	jmp	.LBB0_2515
.LBB0_2518:
	imulq	$165888, -48(%rbp), %rdi
	orq	$64, %rdi
	movq	%rdi, -128(%rbp)
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -72(%rbp)
	jmp	.LBB0_2519
	.p2align	4, 0x90
.LBB0_2529:
	incq	%rcx
	addq	$165888, %rdx
.LBB0_2519:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2530
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2521
	.p2align	4, 0x90
.LBB0_2528:
	incq	%rdi
	addq	$324, %rsi
.LBB0_2521:
	cmpq	$511, %rdi
	jg	.LBB0_2529
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_2523
	.p2align	4, 0x90
.LBB0_2527:
	incq	%r9
	addq	$36, %r8
.LBB0_2523:
	cmpq	$8, %r9
	jg	.LBB0_2528
	xorl	%r10d, %r10d
	cmpq	$8, %r10
	jg	.LBB0_2527
	.p2align	4, 0x90
.LBB0_2526:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$8, %r10
	jle	.LBB0_2526
	jmp	.LBB0_2527
.LBB0_2530:
	movq	%rsp, %r15
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$7, %r8d
	movq	%r8, -56(%rdx)
	movl	$512, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%r14, -88(%rdx)
	movq	-112(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$25088, %edi
	movq	%rdi, -40(%rdx)
	movl	$49, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$9, -24(%rdx)
	movq	$81, -32(%rdx)
	movq	$41472, -40(%rdx)
	movq	$7, -48(%rdx)
	movq	$7, -56(%rdx)
	movq	$512, -64(%rdx)
	movq	$10, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r15, %rsp
	movq	%r14, %rcx
	movq	6216(%rbp), %r10
	jmp	.LBB0_2531
	.p2align	4, 0x90
.LBB0_2541:
	incq	%rax
	addq	$100352, %rcx
.LBB0_2531:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2542
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2533
	.p2align	4, 0x90
.LBB0_2540:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2533:
	cmpq	$511, %rsi
	jg	.LBB0_2541
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2535
	.p2align	4, 0x90
.LBB0_2539:
	incq	%r8
	addq	$28, %rdi
.LBB0_2535:
	cmpq	$6, %r8
	jg	.LBB0_2540
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2539
	.p2align	4, 0x90
.LBB0_2538:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2538
	jmp	.LBB0_2539
.LBB0_2542:
	xorl	%eax, %eax
	jmp	.LBB0_2543
	.p2align	4, 0x90
.LBB0_2562:
	incq	%rax
	addq	$165888, -72(%rbp)
.LBB0_2543:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2563
	movq	6344(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_2545
	.p2align	4, 0x90
.LBB0_2561:
	incq	%rdx
	addq	$18432, -56(%rbp)
.LBB0_2545:
	cmpq	$511, %rdx
	jg	.LBB0_2562
	movq	-72(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_2547
	.p2align	4, 0x90
.LBB0_2560:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$36, %rcx
.LBB0_2547:
	cmpq	$6, %rdi
	jg	.LBB0_2561
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_2549
	.p2align	4, 0x90
.LBB0_2559:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_2549:
	cmpq	$6, %r9
	jg	.LBB0_2560
	movq	%rcx, -80(%rbp)
	movq	%rcx, %rbx
	movq	-56(%rbp), %rcx
	xorl	%r15d, %r15d
	jmp	.LBB0_2551
	.p2align	4, 0x90
.LBB0_2558:
	incq	%r15
	addq	$36, %rcx
	movq	-88(%rbp), %rbx
	addq	$324, %rbx
.LBB0_2551:
	cmpq	$511, %r15
	jg	.LBB0_2559
	movq	%rbx, -88(%rbp)
	movq	%rcx, %r11
	xorl	%esi, %esi
	jmp	.LBB0_2553
	.p2align	4, 0x90
.LBB0_2557:
	incq	%rsi
	addq	$12, %r11
	addq	$36, %rbx
.LBB0_2553:
	cmpq	$2, %rsi
	jg	.LBB0_2558
	xorl	%r13d, %r13d
	cmpq	$2, %r13
	jg	.LBB0_2557
	.p2align	4, 0x90
.LBB0_2556:
	movss	(%rbx,%r13,4), %xmm0
	imulq	$25088, %rax, %r8
	imulq	$49, %rdx, %r10
	addq	%r8, %r10
	leaq	(,%rdi,8), %r8
	subq	%rdi, %r8
	addq	%r10, %r8
	addq	%r9, %r8
	mulss	(%r11,%r13,4), %xmm0
	addss	(%r14,%r8,4), %xmm0
	movss	%xmm0, (%r14,%r8,4)
	incq	%r13
	cmpq	$2, %r13
	jle	.LBB0_2556
	jmp	.LBB0_2557
.LBB0_2563:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r14, %rcx
	jmp	.LBB0_2564
	.p2align	4, 0x90
.LBB0_2574:
	incq	%rax
	addq	$100352, %rcx
.LBB0_2564:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2575
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2566
	.p2align	4, 0x90
.LBB0_2573:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2566:
	cmpq	$511, %rsi
	jg	.LBB0_2574
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2568
	.p2align	4, 0x90
.LBB0_2572:
	incq	%r8
	addq	$28, %rdi
.LBB0_2568:
	cmpq	$6, %r8
	jg	.LBB0_2573
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2572
	.p2align	4, 0x90
.LBB0_2571:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2571
	jmp	.LBB0_2572
.LBB0_2575:
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_2576
	.p2align	4, 0x90
.LBB0_2586:
	incq	%rax
	addq	$401408, %rcx
.LBB0_2576:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2587
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2578
	.p2align	4, 0x90
.LBB0_2585:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2578:
	cmpq	$2047, %rsi
	jg	.LBB0_2586
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2580
	.p2align	4, 0x90
.LBB0_2584:
	incq	%r8
	addq	$28, %rdi
.LBB0_2580:
	cmpq	$6, %r8
	jg	.LBB0_2585
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2584
	.p2align	4, 0x90
.LBB0_2583:
	movq	5960(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2583
	jmp	.LBB0_2584
.LBB0_2587:
	xorl	%eax, %eax
	jmp	.LBB0_2588
	.p2align	4, 0x90
.LBB0_2607:
	incq	%rax
	addq	$100352, %r14
.LBB0_2588:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2608
	movq	6088(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_2590
	.p2align	4, 0x90
.LBB0_2606:
	incq	%rdx
	addq	$2048, -56(%rbp)
.LBB0_2590:
	cmpq	$2047, %rdx
	jg	.LBB0_2607
	movq	%r14, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_2592
	.p2align	4, 0x90
.LBB0_2605:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$28, %rcx
.LBB0_2592:
	cmpq	$6, %rdi
	jg	.LBB0_2606
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_2594
	.p2align	4, 0x90
.LBB0_2604:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_2594:
	cmpq	$6, %r9
	jg	.LBB0_2605
	movq	%rcx, -80(%rbp)
	movq	%rcx, %rbx
	movq	-56(%rbp), %r11
	xorl	%r15d, %r15d
	jmp	.LBB0_2596
	.p2align	4, 0x90
.LBB0_2603:
	incq	%r15
	addq	$4, %r11
	movq	-88(%rbp), %rbx
	addq	$196, %rbx
.LBB0_2596:
	cmpq	$511, %r15
	jg	.LBB0_2604
	movq	%rbx, -88(%rbp)
	movq	%r11, %r13
	xorl	%esi, %esi
	jmp	.LBB0_2598
	.p2align	4, 0x90
.LBB0_2602:
	incq	%rsi
	addq	$4, %r13
	addq	$28, %rbx
.LBB0_2598:
	testq	%rsi, %rsi
	jg	.LBB0_2603
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_2602
	.p2align	4, 0x90
.LBB0_2601:
	movss	(%rbx,%rcx,4), %xmm0
	imulq	$100352, %rax, %r8
	imulq	$49, %rdx, %r10
	addq	%r8, %r10
	leaq	(,%rdi,8), %r8
	subq	%rdi, %r8
	addq	%r10, %r8
	addq	%r9, %r8
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r12,%r8,4), %xmm0
	movss	%xmm0, (%r12,%r8,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_2601
	jmp	.LBB0_2602
.LBB0_2608:
	xorl	%eax, %eax
	movq	%r12, %rcx
	movq	-96(%rbp), %rbx
	jmp	.LBB0_2609
	.p2align	4, 0x90
.LBB0_2619:
	incq	%rax
	addq	$401408, %rbx
	addq	$401408, %rcx
.LBB0_2609:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2620
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2611
	.p2align	4, 0x90
.LBB0_2618:
	incq	%rdi
	addq	$196, %rsi
	addq	$196, %rdx
.LBB0_2611:
	cmpq	$2047, %rdi
	jg	.LBB0_2619
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2613
	.p2align	4, 0x90
.LBB0_2617:
	incq	%r10
	addq	$28, %r9
	addq	$28, %r8
.LBB0_2613:
	cmpq	$6, %r10
	jg	.LBB0_2618
	xorl	%r11d, %r11d
	cmpq	$6, %r11
	jg	.LBB0_2617
	.p2align	4, 0x90
.LBB0_2616:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$6, %r11
	jle	.LBB0_2616
	jmp	.LBB0_2617
.LBB0_2620:
	movq	-136(%rbp), %rbx
	movq	%rbx, %rdi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	movq	%rbx, %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ebx, %ebx
	xorps	%xmm0, %xmm0
	movq	%rax, %rcx
	jmp	.LBB0_2621
	.p2align	4, 0x90
.LBB0_2631:
	incq	%rbx
	addq	$401408, %rcx
	addq	$401408, %r12
.LBB0_2621:
	cmpq	-48(%rbp), %rbx
	jge	.LBB0_2632
	movq	%r12, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2623
	.p2align	4, 0x90
.LBB0_2630:
	incq	%rdi
	addq	$196, %rsi
	addq	$196, %rdx
.LBB0_2623:
	cmpq	$2047, %rdi
	jg	.LBB0_2631
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2625
	.p2align	4, 0x90
.LBB0_2629:
	incq	%r10
	addq	$28, %r9
	addq	$28, %r8
.LBB0_2625:
	cmpq	$6, %r10
	jg	.LBB0_2630
	xorl	%r11d, %r11d
	cmpq	$6, %r11
	jg	.LBB0_2629
	.p2align	4, 0x90
.LBB0_2628:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$6, %r11
	jle	.LBB0_2628
	jmp	.LBB0_2629
.LBB0_2632:
	movq	%rax, -96(%rbp)
	movq	-104(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, -104(%rbp)
	leaq	63(%rax), %r14
	andq	$-64, %r14
	xorl	%eax, %eax
	movq	%r14, %rcx
	movq	5704(%rbp), %r10
	jmp	.LBB0_2633
	.p2align	4, 0x90
.LBB0_2643:
	incq	%rax
	addq	$100352, %rcx
.LBB0_2633:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2644
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2635
	.p2align	4, 0x90
.LBB0_2642:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2635:
	cmpq	$511, %rsi
	jg	.LBB0_2643
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2637
	.p2align	4, 0x90
.LBB0_2641:
	incq	%r8
	addq	$28, %rdi
.LBB0_2637:
	cmpq	$6, %r8
	jg	.LBB0_2642
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2641
	.p2align	4, 0x90
.LBB0_2640:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2640
	jmp	.LBB0_2641
.LBB0_2644:
	xorl	%eax, %eax
	movq	-96(%rbp), %rcx
	movq	%rcx, -72(%rbp)
	jmp	.LBB0_2645
	.p2align	4, 0x90
.LBB0_2664:
	incq	%rax
	addq	$401408, -72(%rbp)
.LBB0_2645:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2665
	movq	5832(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_2647
	.p2align	4, 0x90
.LBB0_2663:
	incq	%rsi
	addq	$8192, -56(%rbp)
.LBB0_2647:
	cmpq	$511, %rsi
	jg	.LBB0_2664
	movq	-72(%rbp), %r11
	xorl	%r8d, %r8d
	jmp	.LBB0_2649
	.p2align	4, 0x90
.LBB0_2662:
	incq	%r8
	movq	-64(%rbp), %r11
	addq	$28, %r11
.LBB0_2649:
	cmpq	$6, %r8
	jg	.LBB0_2663
	movq	%r11, -64(%rbp)
	xorl	%r10d, %r10d
	jmp	.LBB0_2651
	.p2align	4, 0x90
.LBB0_2661:
	incq	%r10
	movq	-80(%rbp), %r11
	addq	$4, %r11
.LBB0_2651:
	cmpq	$6, %r10
	jg	.LBB0_2662
	movq	%r11, -80(%rbp)
	movq	-56(%rbp), %rdx
	xorl	%r12d, %r12d
	jmp	.LBB0_2653
	.p2align	4, 0x90
.LBB0_2660:
	incq	%r12
	addq	$4, %rdx
	movq	-88(%rbp), %r11
	addq	$196, %r11
.LBB0_2653:
	cmpq	$2047, %r12
	jg	.LBB0_2661
	movq	%r11, -88(%rbp)
	movq	%rdx, %r13
	xorl	%ecx, %ecx
	jmp	.LBB0_2655
	.p2align	4, 0x90
.LBB0_2659:
	incq	%rcx
	addq	$4, %r13
	addq	$28, %r11
.LBB0_2655:
	testq	%rcx, %rcx
	jg	.LBB0_2660
	xorl	%ebx, %ebx
	testq	%rbx, %rbx
	jg	.LBB0_2659
	.p2align	4, 0x90
.LBB0_2658:
	movss	(%r11,%rbx,4), %xmm0
	imulq	$25088, %rax, %rdi
	imulq	$49, %rsi, %r9
	addq	%rdi, %r9
	leaq	(,%r8,8), %rdi
	subq	%r8, %rdi
	addq	%r9, %rdi
	addq	%r10, %rdi
	mulss	(%r13,%rbx,4), %xmm0
	addss	(%r14,%rdi,4), %xmm0
	movss	%xmm0, (%r14,%rdi,4)
	incq	%rbx
	testq	%rbx, %rbx
	jle	.LBB0_2658
	jmp	.LBB0_2659
.LBB0_2665:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r14, %rcx
	jmp	.LBB0_2666
	.p2align	4, 0x90
.LBB0_2676:
	incq	%rax
	addq	$100352, %rcx
.LBB0_2666:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2677
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2668
	.p2align	4, 0x90
.LBB0_2675:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2668:
	cmpq	$511, %rsi
	jg	.LBB0_2676
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2670
	.p2align	4, 0x90
.LBB0_2674:
	incq	%r8
	addq	$28, %rdi
.LBB0_2670:
	cmpq	$6, %r8
	jg	.LBB0_2675
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2674
	.p2align	4, 0x90
.LBB0_2673:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2673
	jmp	.LBB0_2674
.LBB0_2677:
	movq	-128(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -72(%rbp)
	jmp	.LBB0_2678
	.p2align	4, 0x90
.LBB0_2688:
	incq	%rcx
	addq	$165888, %rdx
.LBB0_2678:
	cmpq	-48(%rbp), %rcx
	jge	.LBB0_2689
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2680
	.p2align	4, 0x90
.LBB0_2687:
	incq	%rdi
	addq	$324, %rsi
.LBB0_2680:
	cmpq	$511, %rdi
	jg	.LBB0_2688
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_2682
	.p2align	4, 0x90
.LBB0_2686:
	incq	%r9
	addq	$36, %r8
.LBB0_2682:
	cmpq	$8, %r9
	jg	.LBB0_2687
	xorl	%r10d, %r10d
	cmpq	$8, %r10
	jg	.LBB0_2686
	.p2align	4, 0x90
.LBB0_2685:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$8, %r10
	jle	.LBB0_2685
	jmp	.LBB0_2686
.LBB0_2689:
	movq	%rsp, %r12
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$7, %r8d
	movq	%r8, -56(%rdx)
	movl	$512, %esi
	movq	%rsi, -64(%rdx)
	movq	-48(%rbp), %rsi
	movq	%rsi, -72(%rdx)
	movq	%r14, -88(%rdx)
	movq	-104(%rbp), %rdi
	movq	%rdi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$25088, %edi
	movq	%rdi, -40(%rdx)
	movl	$49, %edi
	movq	%rdi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movl	$1, %edi
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	%rsi, -72(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$9, -24(%rdx)
	movq	$81, -32(%rdx)
	movq	$41472, -40(%rdx)
	movq	$7, -48(%rdx)
	movq	$7, -56(%rdx)
	movq	$512, -64(%rdx)
	movq	$10, -80(%rdx)
	movq	%rsp, %rax
	leaq	-16(%rax), %rsi
	movq	%rsi, %rsp
	movq	%rcx, -8(%rax)
	movq	$4, -16(%rax)
	movq	%rsp, %rax
	leaq	-16(%rax), %rdx
	movq	%rdx, %rsp
	movq	%rdi, -8(%rax)
	movq	$4, -16(%rax)
	movl	$4, %edi
	callq	memrefCopy@PLT
	xorl	%eax, %eax
	movq	%r12, %rsp
	movq	%r14, %rcx
	movq	5448(%rbp), %r10
	jmp	.LBB0_2690
	.p2align	4, 0x90
.LBB0_2700:
	incq	%rax
	addq	$100352, %rcx
.LBB0_2690:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2701
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2692
	.p2align	4, 0x90
.LBB0_2699:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2692:
	cmpq	$511, %rsi
	jg	.LBB0_2700
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2694
	.p2align	4, 0x90
.LBB0_2698:
	incq	%r8
	addq	$28, %rdi
.LBB0_2694:
	cmpq	$6, %r8
	jg	.LBB0_2699
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2698
	.p2align	4, 0x90
.LBB0_2697:
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2697
	jmp	.LBB0_2698
.LBB0_2701:
	xorl	%eax, %eax
	jmp	.LBB0_2702
	.p2align	4, 0x90
.LBB0_2721:
	incq	%rax
	addq	$165888, -72(%rbp)
.LBB0_2702:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2722
	movq	5576(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_2704
	.p2align	4, 0x90
.LBB0_2720:
	incq	%rdx
	addq	$18432, -56(%rbp)
.LBB0_2704:
	cmpq	$511, %rdx
	jg	.LBB0_2721
	movq	-72(%rbp), %rcx
	xorl	%edi, %edi
	jmp	.LBB0_2706
	.p2align	4, 0x90
.LBB0_2719:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$36, %rcx
.LBB0_2706:
	cmpq	$6, %rdi
	jg	.LBB0_2720
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_2708
	.p2align	4, 0x90
.LBB0_2718:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_2708:
	cmpq	$6, %r9
	jg	.LBB0_2719
	movq	%rcx, -80(%rbp)
	movq	%rcx, %rbx
	movq	-56(%rbp), %rcx
	xorl	%r12d, %r12d
	jmp	.LBB0_2710
	.p2align	4, 0x90
.LBB0_2717:
	incq	%r12
	addq	$36, %rcx
	movq	-88(%rbp), %rbx
	addq	$324, %rbx
.LBB0_2710:
	cmpq	$511, %r12
	jg	.LBB0_2718
	movq	%rbx, -88(%rbp)
	movq	%rcx, %r11
	xorl	%esi, %esi
	jmp	.LBB0_2712
	.p2align	4, 0x90
.LBB0_2716:
	incq	%rsi
	addq	$12, %r11
	addq	$36, %rbx
.LBB0_2712:
	cmpq	$2, %rsi
	jg	.LBB0_2717
	xorl	%r13d, %r13d
	cmpq	$2, %r13
	jg	.LBB0_2716
	.p2align	4, 0x90
.LBB0_2715:
	movss	(%rbx,%r13,4), %xmm0
	imulq	$25088, %rax, %r8
	imulq	$49, %rdx, %r10
	addq	%r8, %r10
	leaq	(,%rdi,8), %r8
	subq	%rdi, %r8
	addq	%r10, %r8
	addq	%r9, %r8
	mulss	(%r11,%r13,4), %xmm0
	addss	(%r14,%r8,4), %xmm0
	movss	%xmm0, (%r14,%r8,4)
	incq	%r13
	cmpq	$2, %r13
	jle	.LBB0_2715
	jmp	.LBB0_2716
.LBB0_2722:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%r14, %rcx
	jmp	.LBB0_2723
	.p2align	4, 0x90
.LBB0_2733:
	incq	%rax
	addq	$100352, %rcx
.LBB0_2723:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2734
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2725
	.p2align	4, 0x90
.LBB0_2732:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2725:
	cmpq	$511, %rsi
	jg	.LBB0_2733
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2727
	.p2align	4, 0x90
.LBB0_2731:
	incq	%r8
	addq	$28, %rdi
.LBB0_2727:
	cmpq	$6, %r8
	jg	.LBB0_2732
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2731
	.p2align	4, 0x90
.LBB0_2730:
	movss	(%rdi,%r9,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2730
	jmp	.LBB0_2731
.LBB0_2734:
	xorl	%eax, %eax
	movq	%r15, %rcx
	jmp	.LBB0_2735
	.p2align	4, 0x90
.LBB0_2745:
	incq	%rax
	addq	$401408, %rcx
.LBB0_2735:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2746
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2737
	.p2align	4, 0x90
.LBB0_2744:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2737:
	cmpq	$2047, %rsi
	jg	.LBB0_2745
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2739
	.p2align	4, 0x90
.LBB0_2743:
	incq	%r8
	addq	$28, %rdi
.LBB0_2739:
	cmpq	$6, %r8
	jg	.LBB0_2744
	xorl	%r9d, %r9d
	cmpq	$6, %r9
	jg	.LBB0_2743
	.p2align	4, 0x90
.LBB0_2742:
	movq	5192(%rbp), %r10
	movss	(%r10,%rsi,4), %xmm0
	movss	%xmm0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$6, %r9
	jle	.LBB0_2742
	jmp	.LBB0_2743
.LBB0_2746:
	xorl	%eax, %eax
	jmp	.LBB0_2747
	.p2align	4, 0x90
.LBB0_2766:
	incq	%rax
	addq	$100352, %r14
.LBB0_2747:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2767
	movq	5320(%rbp), %rcx
	movq	%rcx, -56(%rbp)
	xorl	%edx, %edx
	jmp	.LBB0_2749
	.p2align	4, 0x90
.LBB0_2765:
	incq	%rdx
	addq	$2048, -56(%rbp)
.LBB0_2749:
	cmpq	$2047, %rdx
	jg	.LBB0_2766
	movq	%r14, %rcx
	xorl	%edi, %edi
	jmp	.LBB0_2751
	.p2align	4, 0x90
.LBB0_2764:
	incq	%rdi
	movq	-64(%rbp), %rcx
	addq	$28, %rcx
.LBB0_2751:
	cmpq	$6, %rdi
	jg	.LBB0_2765
	movq	%rcx, -64(%rbp)
	xorl	%r9d, %r9d
	jmp	.LBB0_2753
	.p2align	4, 0x90
.LBB0_2763:
	incq	%r9
	movq	-80(%rbp), %rcx
	addq	$4, %rcx
.LBB0_2753:
	cmpq	$6, %r9
	jg	.LBB0_2764
	movq	%rcx, -80(%rbp)
	movq	%rcx, %rbx
	movq	-56(%rbp), %r11
	xorl	%r12d, %r12d
	jmp	.LBB0_2755
	.p2align	4, 0x90
.LBB0_2762:
	incq	%r12
	addq	$4, %r11
	movq	-88(%rbp), %rbx
	addq	$196, %rbx
.LBB0_2755:
	cmpq	$511, %r12
	jg	.LBB0_2763
	movq	%rbx, -88(%rbp)
	movq	%r11, %r13
	xorl	%esi, %esi
	jmp	.LBB0_2757
	.p2align	4, 0x90
.LBB0_2761:
	incq	%rsi
	addq	$4, %r13
	addq	$28, %rbx
.LBB0_2757:
	testq	%rsi, %rsi
	jg	.LBB0_2762
	xorl	%ecx, %ecx
	testq	%rcx, %rcx
	jg	.LBB0_2761
	.p2align	4, 0x90
.LBB0_2760:
	movss	(%rbx,%rcx,4), %xmm0
	imulq	$100352, %rax, %r8
	imulq	$49, %rdx, %r10
	addq	%r8, %r10
	leaq	(,%rdi,8), %r8
	subq	%rdi, %r8
	addq	%r10, %r8
	addq	%r9, %r8
	mulss	(%r13,%rcx,4), %xmm0
	addss	(%r15,%r8,4), %xmm0
	movss	%xmm0, (%r15,%r8,4)
	incq	%rcx
	testq	%rcx, %rcx
	jle	.LBB0_2760
	jmp	.LBB0_2761
.LBB0_2767:
	xorl	%eax, %eax
	movq	%r15, %rcx
	movq	-96(%rbp), %rbx
	jmp	.LBB0_2768
	.p2align	4, 0x90
.LBB0_2778:
	incq	%rax
	addq	$401408, %rbx
	addq	$401408, %rcx
.LBB0_2768:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2779
	movq	%rcx, %rdx
	movq	%rbx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2770
	.p2align	4, 0x90
.LBB0_2777:
	incq	%rdi
	addq	$196, %rsi
	addq	$196, %rdx
.LBB0_2770:
	cmpq	$2047, %rdi
	jg	.LBB0_2778
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2772
	.p2align	4, 0x90
.LBB0_2776:
	incq	%r10
	addq	$28, %r9
	addq	$28, %r8
.LBB0_2772:
	cmpq	$6, %r10
	jg	.LBB0_2777
	xorl	%r11d, %r11d
	cmpq	$6, %r11
	jg	.LBB0_2776
	.p2align	4, 0x90
.LBB0_2775:
	movss	(%r8,%r11,4), %xmm0
	addss	(%r9,%r11,4), %xmm0
	movss	%xmm0, (%r8,%r11,4)
	incq	%r11
	cmpq	$6, %r11
	jle	.LBB0_2775
	jmp	.LBB0_2776
.LBB0_2779:
	movq	-136(%rbp), %rdi
	callq	malloc@PLT
	movq	%rax, %rbx
	addq	$63, %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	%rbx, %rcx
	jmp	.LBB0_2780
	.p2align	4, 0x90
.LBB0_2790:
	incq	%rax
	addq	$401408, %rcx
	addq	$401408, %r15
.LBB0_2780:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2791
	movq	%r15, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2782
	.p2align	4, 0x90
.LBB0_2789:
	incq	%rdi
	addq	$196, %rsi
	addq	$196, %rdx
.LBB0_2782:
	cmpq	$2047, %rdi
	jg	.LBB0_2790
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2784
	.p2align	4, 0x90
.LBB0_2788:
	incq	%r10
	addq	$28, %r9
	addq	$28, %r8
.LBB0_2784:
	cmpq	$6, %r10
	jg	.LBB0_2789
	xorl	%r11d, %r11d
	cmpq	$6, %r11
	jg	.LBB0_2788
	.p2align	4, 0x90
.LBB0_2787:
	movss	(%r8,%r11,4), %xmm1
	movaps	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r9,%r11,4)
	incq	%r11
	cmpq	$6, %r11
	jle	.LBB0_2787
	jmp	.LBB0_2788
.LBB0_2791:
	movq	-48(%rbp), %r12
	shlq	$13, %r12
	orq	$64, %r12
	movq	%r12, %rdi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	xorl	%eax, %eax
	movq	%r15, %rcx
	jmp	.LBB0_2792
	.p2align	4, 0x90
.LBB0_2802:
	incq	%rax
	addq	$8192, %rcx
.LBB0_2792:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2803
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2794
	.p2align	4, 0x90
.LBB0_2801:
	incq	%rsi
	addq	$4, %rdx
.LBB0_2794:
	cmpq	$2047, %rsi
	jg	.LBB0_2802
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_2796
	.p2align	4, 0x90
.LBB0_2800:
	incq	%r8
	addq	$4, %rdi
.LBB0_2796:
	testq	%r8, %r8
	jg	.LBB0_2801
	xorl	%r9d, %r9d
	testq	%r9, %r9
	jg	.LBB0_2800
	.p2align	4, 0x90
.LBB0_2799:
	movl	$0, (%rdi,%r9,4)
	incq	%r9
	testq	%r9, %r9
	jle	.LBB0_2799
	jmp	.LBB0_2800
.LBB0_2803:
	xorl	%eax, %eax
	jmp	.LBB0_2804
	.p2align	4, 0x90
.LBB0_2814:
	incq	%rax
	addq	$401408, %rbx
.LBB0_2804:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2815
	movq	%rax, %rcx
	shlq	$11, %rcx
	movq	%rbx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2806
	.p2align	4, 0x90
.LBB0_2813:
	incq	%rsi
	addq	$196, %rdx
.LBB0_2806:
	cmpq	$2047, %rsi
	jg	.LBB0_2814
	leaq	(%rcx,%rsi), %rdi
	movq	%rdx, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_2808
	.p2align	4, 0x90
.LBB0_2812:
	incq	%r9
	addq	$28, %r8
.LBB0_2808:
	cmpq	$6, %r9
	jg	.LBB0_2813
	xorl	%r10d, %r10d
	cmpq	$6, %r10
	jg	.LBB0_2812
	.p2align	4, 0x90
.LBB0_2811:
	movss	(%r8,%r10,4), %xmm0
	addss	(%r15,%rdi,4), %xmm0
	movss	%xmm0, (%r15,%rdi,4)
	incq	%r10
	cmpq	$6, %r10
	jle	.LBB0_2811
	jmp	.LBB0_2812
.LBB0_2815:
	movq	%r12, %rdi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	xorl	%eax, %eax
	movss	.LCPI0_0(%rip), %xmm0
	movq	%r12, %rcx
	jmp	.LBB0_2816
	.p2align	4, 0x90
.LBB0_2826:
	incq	%rax
	addq	$8192, %rcx
	addq	$8192, %r15
.LBB0_2816:
	cmpq	-48(%rbp), %rax
	jge	.LBB0_2827
	movq	%r15, %rdx
	movq	%rcx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_2818
	.p2align	4, 0x90
.LBB0_2825:
	incq	%rdi
	addq	$4, %rsi
	addq	$4, %rdx
.LBB0_2818:
	cmpq	$2047, %rdi
	jg	.LBB0_2826
	movq	%rdx, %r8
	movq	%rsi, %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_2820
	.p2align	4, 0x90
.LBB0_2824:
	incq	%r10
	addq	$4, %r9
	addq	$4, %r8
.LBB0_2820:
	testq	%r10, %r10
	jg	.LBB0_2825
	xorl	%r11d, %r11d
	testq	%r11, %r11
	jg	.LBB0_2824
	.p2align	4, 0x90
.LBB0_2823:
	movss	(%r8,%r11,4), %xmm1
	mulss	%xmm0, %xmm1
	movss	%xmm1, (%r9,%r11,4)
	incq	%r11
	testq	%r11, %r11
	jle	.LBB0_2823
	jmp	.LBB0_2824
.LBB0_2827:
	movq	3072(%rbp), %rdx
	movq	24(%rdx), %r15
	movq	8(%rdx), %r13
	movq	16(%rdx), %rax
	movq	%r15, %rcx
	imulq	%rax, %rcx
	movq	%rcx, %r14
	imulq	%r13, %r14
	imulq	(%rdx), %r13
	imulq	%rax, %r13
	cmpq	%rcx, %r14
	cmovgeq	%rcx, %r14
	cmpq	%r15, %r14
	cmovgeq	%r15, %r14
	imulq	$4004, %r13, %rdi
	addq	$64, %rdi
	movq	%rdi, -88(%rbp)
	callq	malloc@PLT
	movq	%rax, -64(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	movq	1440(%rbp), %r10
	jmp	.LBB0_2828
	.p2align	4, 0x90
.LBB0_2832:
	incq	%rax
	addq	$4004, %rcx
.LBB0_2828:
	cmpq	%r13, %rax
	jge	.LBB0_2833
	xorl	%edx, %edx
	cmpq	$1000, %rdx
	jg	.LBB0_2832
	.p2align	4, 0x90
.LBB0_2831:
	movl	$0, (%rcx,%rdx,4)
	incq	%rdx
	cmpq	$1000, %rdx
	jle	.LBB0_2831
	jmp	.LBB0_2832
.LBB0_2833:
	shlq	$2, %r14
	xorl	%eax, %eax
	jmp	.LBB0_2834
	.p2align	4, 0x90
.LBB0_2841:
	incq	%rax
	addq	%r14, %r12
.LBB0_2834:
	cmpq	%r13, %rax
	jge	.LBB0_2842
	imulq	$1001, %rax, %rcx
	movq	%r10, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_2836
	.p2align	4, 0x90
.LBB0_2840:
	incq	%rsi
	addq	$4, %rdx
.LBB0_2836:
	cmpq	$1000, %rsi
	jg	.LBB0_2841
	leaq	(%rcx,%rsi), %rdi
	movq	%rdx, %r8
	xorl	%r9d, %r9d
	cmpq	%r15, %r9
	jge	.LBB0_2840
	.p2align	4, 0x90
.LBB0_2839:
	movss	(%r12,%r9,4), %xmm0
	mulss	(%r8), %xmm0
	addss	(%rbx,%rdi,4), %xmm0
	movss	%xmm0, (%rbx,%rdi,4)
	incq	%r9
	addq	$4004, %r8
	cmpq	%r15, %r9
	jl	.LBB0_2839
	jmp	.LBB0_2840
.LBB0_2842:
	movq	-88(%rbp), %rdi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%r8d, %r8d
	movq	%rbx, %rcx
	movq	%rax, %rdx
	movq	1536(%rbp), %rdi
	jmp	.LBB0_2843
	.p2align	4, 0x90
.LBB0_2847:
	incq	%r8
	addq	$4004, %rdx
	addq	$4004, %rcx
.LBB0_2843:
	cmpq	%r13, %r8
	jge	.LBB0_2848
	xorl	%esi, %esi
	cmpq	$1000, %rsi
	jg	.LBB0_2847
	.p2align	4, 0x90
.LBB0_2846:
	movss	(%rcx,%rsi,4), %xmm0
	addss	(%rdi,%rsi,4), %xmm0
	movss	%xmm0, (%rdx,%rsi,4)
	incq	%rsi
	cmpq	$1000, %rsi
	jle	.LBB0_2846
	jmp	.LBB0_2847
.LBB0_2848:
	movq	%rax, -56(%rbp)
	leaq	64(,%r13,4), %rdi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	xorl	%eax, %eax
	cmpq	%r13, %rax
	jge	.LBB0_2850
	.p2align	4, 0x90
.LBB0_2889:
	movl	$-8388608, (%r15,%rax,4)
	incq	%rax
	cmpq	%r13, %rax
	jl	.LBB0_2889
.LBB0_2850:
	xorl	%eax, %eax
	movq	-56(%rbp), %rcx
	jmp	.LBB0_2851
	.p2align	4, 0x90
.LBB0_2859:
	incq	%rax
	addq	$4004, %rcx
.LBB0_2851:
	cmpq	%r13, %rax
	jge	.LBB0_2860
	xorl	%edx, %edx
	jmp	.LBB0_2853
	.p2align	4, 0x90
.LBB0_2858:
	movdqa	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r15,%rax,4)
	incq	%rdx
.LBB0_2853:
	cmpq	$1000, %rdx
	jg	.LBB0_2859
	movd	(%rcx,%rdx,4), %xmm2
	movd	(%r15,%rax,4), %xmm0
	movd	%xmm2, %esi
	testl	%esi, %esi
	movdqa	%xmm2, %xmm1
	js	.LBB0_2856
	movdqa	%xmm0, %xmm1
.LBB0_2856:
	js	.LBB0_2858
	movdqa	%xmm2, %xmm0
	jmp	.LBB0_2858
.LBB0_2860:
	movq	%r13, -80(%rbp)
	xorl	%r14d, %r14d
	movq	-56(%rbp), %rcx
	movq	%rcx, -88(%rbp)
	movq	%rbx, %r13
	jmp	.LBB0_2861
	.p2align	4, 0x90
.LBB0_2865:
	incq	%r14
	addq	$4004, %r13
	addq	$4004, -88(%rbp)
.LBB0_2861:
	cmpq	-80(%rbp), %r14
	jge	.LBB0_2866
	xorl	%r12d, %r12d
	cmpq	$1000, %r12
	jg	.LBB0_2865
	.p2align	4, 0x90
.LBB0_2864:
	movq	-88(%rbp), %rcx
	movss	(%rcx,%r12,4), %xmm0
	subss	(%r15,%r14,4), %xmm0
	callq	expf@PLT
	movss	%xmm0, (%r13,%r12,4)
	incq	%r12
	cmpq	$1000, %r12
	jle	.LBB0_2864
	jmp	.LBB0_2865
.LBB0_2866:
	xorl	%eax, %eax
	movq	-80(%rbp), %r14
	cmpq	%r14, %rax
	jge	.LBB0_2868
	.p2align	4, 0x90
.LBB0_2890:
	movl	$0, (%r15,%rax,4)
	incq	%rax
	cmpq	%r14, %rax
	jl	.LBB0_2890
.LBB0_2868:
	xorl	%eax, %eax
	movq	%rbx, %rcx
	jmp	.LBB0_2869
	.p2align	4, 0x90
.LBB0_2873:
	incq	%rax
	addq	$4004, %rcx
.LBB0_2869:
	cmpq	%r14, %rax
	jge	.LBB0_2874
	xorl	%edx, %edx
	cmpq	$1000, %rdx
	jg	.LBB0_2873
	.p2align	4, 0x90
.LBB0_2872:
	movss	(%rcx,%rdx,4), %xmm0
	addss	(%r15,%rax,4), %xmm0
	movss	%xmm0, (%r15,%rax,4)
	incq	%rdx
	cmpq	$1000, %rdx
	jle	.LBB0_2872
	jmp	.LBB0_2873
.LBB0_2874:
	xorl	%eax, %eax
	movq	%rbx, %rcx
	jmp	.LBB0_2875
	.p2align	4, 0x90
.LBB0_2879:
	incq	%rax
	addq	$4004, %rcx
.LBB0_2875:
	cmpq	%r14, %rax
	jge	.LBB0_2880
	xorl	%edx, %edx
	cmpq	$1000, %rdx
	jg	.LBB0_2879
	.p2align	4, 0x90
.LBB0_2878:
	movss	(%rcx,%rdx,4), %xmm0
	divss	(%r15,%rax,4), %xmm0
	movss	%xmm0, (%rcx,%rdx,4)
	incq	%rdx
	cmpq	$1000, %rdx
	jle	.LBB0_2878
	jmp	.LBB0_2879
.LBB0_2880:
	leaq	64(,%r14,8), %rdi
	callq	malloc@PLT
	leaq	63(%rax), %rdx
	andq	$-64, %rdx
	movl	$1, %ecx
	xorl	%esi, %esi
	movq	-56(%rbp), %r15
	cmpq	%r14, %rsi
	jge	.LBB0_2882
	.p2align	4, 0x90
.LBB0_2891:
	movq	$0, (%rdx,%rsi,8)
	incq	%rsi
	cmpq	%r14, %rsi
	jl	.LBB0_2891
.LBB0_2882:
	xorl	%esi, %esi
	movq	%r15, %rdi
	jmp	.LBB0_2883
	.p2align	4, 0x90
.LBB0_2887:
	incq	%rsi
	addq	$4004, %rdi
.LBB0_2883:
	cmpq	%r14, %rsi
	jge	.LBB0_2888
	imulq	$1001, %rsi, %r8
	xorl	%r9d, %r9d
	cmpq	$1000, %r9
	jg	.LBB0_2887
	.p2align	4, 0x90
.LBB0_2886:
	movss	(%rdi,%r9,4), %xmm0
	movq	(%rdx,%rsi,8), %r10
	leaq	(%r8,%r10), %r11
	ucomiss	(%r15,%r11,4), %xmm0
	cmovaq	%r9, %r10
	movq	%r10, (%rdx,%rsi,8)
	incq	%r9
	cmpq	$1000, %r9
	jle	.LBB0_2886
	jmp	.LBB0_2887
.LBB0_2888:
	movq	-152(%rbp), %rsi
	movq	-64(%rbp), %rdi
	movq	%rdi, 40(%rsi)
	movq	%rax, (%rsi)
	movq	%rbx, 48(%rsi)
	movq	%rdx, 8(%rsi)
	movq	%r14, 64(%rsi)
	movq	%r14, 24(%rsi)
	movl	$1001, %eax
	movq	%rax, 72(%rsi)
	movq	%rcx, 32(%rsi)
	movq	%rax, 80(%rsi)
	movl	$1, %eax
	movq	%rax, 88(%rsi)
	movq	$0, 56(%rsi)
	movq	$0, 16(%rsi)
	movq	%rsi, %rax
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	tensorCompForwardImpl, .Lfunc_end0-tensorCompForwardImpl
	.cfi_endproc

	.section	".note.GNU-stack","",@progbits
