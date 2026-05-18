	.text
	.file	"tensor_network"
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
	subq	$56, %rsp
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	movq	%rsi, -96(%rbp)
	movq	%rdi, %r14
	movl	$1, %ebx
	movl	$28, %r12d
	movl	$784, %r15d
	xorl	%r13d, %r13d
	movl	$4160, %edi
	callq	malloc@PLT
	movq	%rax, %rcx
	addq	$63, %rcx
	andq	$-64, %rcx
	movq	%rcx, -88(%rbp)
	jmp	.LBB0_1
	.p2align	4, 0x90
.LBB0_11:
	incq	%r13
	addq	$4096, %rcx
.LBB0_1:
	testq	%r13, %r13
	jg	.LBB0_12
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_3
	.p2align	4, 0x90
.LBB0_10:
	incq	%rsi
	addq	$4096, %rdx
.LBB0_3:
	testq	%rsi, %rsi
	jg	.LBB0_11
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_5
	.p2align	4, 0x90
.LBB0_9:
	incq	%r8
	subq	$-128, %rdi
.LBB0_5:
	cmpq	$31, %r8
	jg	.LBB0_10
	xorl	%r9d, %r9d
	cmpq	$31, %r9
	jg	.LBB0_9
	.p2align	4, 0x90
.LBB0_8:
	movl	$0, (%rdi,%r9,4)
	incq	%r9
	cmpq	$31, %r9
	jle	.LBB0_8
	jmp	.LBB0_9
.LBB0_12:
	movq	%rsp, %r13
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movq	%r12, -56(%rdx)
	movq	%rbx, -64(%rdx)
	movq	%rbx, -72(%rdx)
	movq	%r14, -88(%rdx)
	movq	%r14, -96(%rdx)
	movq	%r12, -48(%rdx)
	movq	%r15, -40(%rdx)
	movq	%r15, -32(%rdx)
	movq	%r12, -24(%rdx)
	movq	%rbx, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	-88(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$32, -24(%rdx)
	movq	$1024, -32(%rdx)
	movq	$1024, -40(%rdx)
	movq	$28, -48(%rdx)
	movq	$28, -56(%rdx)
	movq	$1, -64(%rdx)
	movq	$1, -72(%rdx)
	movq	$66, -80(%rdx)
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
	movl	$25152, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	movq	%rax, -48(%rbp)
	movl	$25152, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ecx, %ecx
	movq	%rax, %rdx
	jmp	.LBB0_13
	.p2align	4, 0x90
.LBB0_23:
	incq	%rcx
	addq	$25088, %rdx
.LBB0_13:
	testq	%rcx, %rcx
	jg	.LBB0_24
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_15
	.p2align	4, 0x90
.LBB0_22:
	incq	%rdi
	addq	$3136, %rsi
.LBB0_15:
	cmpq	$7, %rdi
	jg	.LBB0_23
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_17
	.p2align	4, 0x90
.LBB0_21:
	incq	%r9
	addq	$112, %r8
.LBB0_17:
	cmpq	$27, %r9
	jg	.LBB0_22
	xorl	%r10d, %r10d
	cmpq	$27, %r10
	jg	.LBB0_21
	.p2align	4, 0x90
.LBB0_20:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$27, %r10
	jle	.LBB0_20
	jmp	.LBB0_21
.LBB0_24:
	xorl	%ecx, %ecx
	jmp	.LBB0_25
	.p2align	4, 0x90
.LBB0_44:
	incq	%rcx
	addq	$4096, -88(%rbp)
.LBB0_25:
	testq	%rcx, %rcx
	jg	.LBB0_45
	leaq	.L__constant_8x1x5x5xf32(%rip), %rdx
	movq	%rdx, -56(%rbp)
	xorl	%edi, %edi
	jmp	.LBB0_27
	.p2align	4, 0x90
.LBB0_43:
	incq	%rdi
	addq	$100, -56(%rbp)
.LBB0_27:
	cmpq	$7, %rdi
	jg	.LBB0_44
	movq	-88(%rbp), %rdx
	xorl	%r9d, %r9d
	jmp	.LBB0_29
	.p2align	4, 0x90
.LBB0_42:
	incq	%r9
	movq	-72(%rbp), %rdx
	subq	$-128, %rdx
.LBB0_29:
	cmpq	$27, %r9
	jg	.LBB0_43
	movq	%rdx, -72(%rbp)
	movq	%rdx, %rbx
	xorl	%r11d, %r11d
	jmp	.LBB0_31
	.p2align	4, 0x90
.LBB0_41:
	incq	%r11
	movq	-64(%rbp), %rbx
	addq	$4, %rbx
.LBB0_31:
	cmpq	$27, %r11
	jg	.LBB0_42
	movq	%rbx, -64(%rbp)
	movq	-56(%rbp), %rsi
	xorl	%r13d, %r13d
	jmp	.LBB0_33
	.p2align	4, 0x90
.LBB0_40:
	incq	%r13
	addq	$100, %rsi
	addq	$4096, %rbx
.LBB0_33:
	testq	%r13, %r13
	jg	.LBB0_41
	movq	%rbx, %r14
	movq	%rsi, %r12
	xorl	%r15d, %r15d
	jmp	.LBB0_35
	.p2align	4, 0x90
.LBB0_39:
	incq	%r15
	addq	$20, %r12
	subq	$-128, %r14
.LBB0_35:
	cmpq	$4, %r15
	jg	.LBB0_40
	xorl	%r8d, %r8d
	cmpq	$4, %r8
	jg	.LBB0_39
	.p2align	4, 0x90
.LBB0_38:
	movss	(%r14,%r8,4), %xmm0
	imulq	$6272, %rcx, %rdx
	imulq	$784, %rdi, %r10
	addq	%rdx, %r10
	leaq	(%r9,%r9,8), %rdx
	leaq	(%rdx,%rdx,2), %rdx
	addq	%r9, %rdx
	addq	%r11, %r10
	addq	%rdx, %r10
	mulss	(%r12,%r8,4), %xmm0
	addss	(%rax,%r10,4), %xmm0
	movss	%xmm0, (%rax,%r10,4)
	incq	%r8
	cmpq	$4, %r8
	jle	.LBB0_38
	jmp	.LBB0_39
.LBB0_45:
	xorl	%ecx, %ecx
	leaq	.L__constant_8x1x1xf32(%rip), %rdx
	movq	-48(%rbp), %rsi
	jmp	.LBB0_46
	.p2align	4, 0x90
.LBB0_56:
	incq	%rcx
	addq	$25088, %rsi
.LBB0_46:
	testq	%rcx, %rcx
	jg	.LBB0_57
	movq	%rax, %rdi
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_48
	.p2align	4, 0x90
.LBB0_55:
	incq	%r9
	addq	$3136, %r8
	addq	$3136, %rdi
.LBB0_48:
	cmpq	$7, %r9
	jg	.LBB0_56
	movq	%rdi, %r10
	movq	%r8, %r11
	xorl	%ebx, %ebx
	jmp	.LBB0_50
	.p2align	4, 0x90
.LBB0_54:
	incq	%rbx
	addq	$112, %r11
	addq	$112, %r10
.LBB0_50:
	cmpq	$27, %rbx
	jg	.LBB0_55
	xorl	%r14d, %r14d
	cmpq	$27, %r14
	jg	.LBB0_54
	.p2align	4, 0x90
.LBB0_53:
	movss	(%r10,%r14,4), %xmm0
	addss	(%rdx,%r9,4), %xmm0
	movss	%xmm0, (%r11,%r14,4)
	incq	%r14
	cmpq	$27, %r14
	jle	.LBB0_53
	jmp	.LBB0_54
.LBB0_57:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	-48(%rbp), %rcx
	jmp	.LBB0_58
	.p2align	4, 0x90
.LBB0_68:
	incq	%rax
	addq	$25088, %rcx
.LBB0_58:
	testq	%rax, %rax
	jg	.LBB0_69
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_60
	.p2align	4, 0x90
.LBB0_67:
	incq	%rsi
	addq	$3136, %rdx
.LBB0_60:
	cmpq	$7, %rsi
	jg	.LBB0_68
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_62
	.p2align	4, 0x90
.LBB0_66:
	incq	%r8
	addq	$112, %rdi
.LBB0_62:
	cmpq	$27, %r8
	jg	.LBB0_67
	xorl	%r9d, %r9d
	cmpq	$27, %r9
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
	cmpq	$27, %r9
	jle	.LBB0_65
	jmp	.LBB0_66
.LBB0_69:
	movl	$6336, %edi
	callq	malloc@PLT
	movq	%rax, -72(%rbp)
	leaq	63(%rax), %rbx
	andq	$-64, %rbx
	xorl	%eax, %eax
	movq	%rbx, %rcx
	jmp	.LBB0_70
	.p2align	4, 0x90
.LBB0_80:
	incq	%rax
	addq	$6272, %rcx
.LBB0_70:
	testq	%rax, %rax
	jg	.LBB0_81
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_72
	.p2align	4, 0x90
.LBB0_79:
	incq	%rsi
	addq	$784, %rdx
.LBB0_72:
	cmpq	$7, %rsi
	jg	.LBB0_80
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_74
	.p2align	4, 0x90
.LBB0_78:
	incq	%r8
	addq	$56, %rdi
.LBB0_74:
	cmpq	$13, %r8
	jg	.LBB0_79
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_78
	.p2align	4, 0x90
.LBB0_77:
	movl	$-8388608, (%rdi,%r9,4)
	incq	%r9
	cmpq	$13, %r9
	jle	.LBB0_77
	jmp	.LBB0_78
.LBB0_81:
	xorl	%eax, %eax
	jmp	.LBB0_82
	.p2align	4, 0x90
.LBB0_102:
	movq	-56(%rbp), %rax
	incq	%rax
	addq	$25088, -48(%rbp)
.LBB0_82:
	testq	%rax, %rax
	jg	.LBB0_103
	movq	%rax, -56(%rbp)
	imulq	$1568, %rax, %rax
	movq	%rax, -64(%rbp)
	movq	-48(%rbp), %r14
	xorl	%esi, %esi
	jmp	.LBB0_84
	.p2align	4, 0x90
.LBB0_101:
	incq	%rsi
	addq	$3136, %r14
.LBB0_84:
	cmpq	$7, %rsi
	jg	.LBB0_102
	imulq	$196, %rsi, %rdi
	addq	-64(%rbp), %rdi
	movq	%r14, %rdx
	xorl	%r9d, %r9d
	jmp	.LBB0_86
	.p2align	4, 0x90
.LBB0_100:
	incq	%r9
	addq	$224, %rdx
.LBB0_86:
	cmpq	$13, %r9
	jg	.LBB0_101
	movq	%r9, %r10
	shlq	$4, %r10
	subq	%r9, %r10
	subq	%r9, %r10
	addq	%rdi, %r10
	movq	%rdx, %r8
	xorl	%r12d, %r12d
	jmp	.LBB0_88
	.p2align	4, 0x90
.LBB0_99:
	incq	%r12
	addq	$8, %r8
.LBB0_88:
	cmpq	$13, %r12
	jg	.LBB0_100
	leaq	(%r10,%r12), %r13
	movq	%r8, %r11
	xorl	%r15d, %r15d
	jmp	.LBB0_90
	.p2align	4, 0x90
.LBB0_98:
	incq	%r15
	addq	$112, %r11
.LBB0_90:
	cmpq	$1, %r15
	jg	.LBB0_99
	xorl	%eax, %eax
	jmp	.LBB0_92
	.p2align	4, 0x90
.LBB0_97:
	movdqa	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%rbx,%r13,4)
	incq	%rax
.LBB0_92:
	cmpq	$1, %rax
	jg	.LBB0_98
	movd	(%r11,%rax,4), %xmm0
	movd	(%rbx,%r13,4), %xmm2
	movd	%xmm2, %ecx
	testl	%ecx, %ecx
	movdqa	%xmm2, %xmm1
	js	.LBB0_95
	movdqa	%xmm0, %xmm1
.LBB0_95:
	js	.LBB0_97
	movdqa	%xmm2, %xmm0
	jmp	.LBB0_97
.LBB0_103:
	movl	$10432, %edi
	callq	malloc@PLT
	movq	%rax, %rdx
	addq	$63, %rdx
	andq	$-64, %rdx
	xorl	%ecx, %ecx
	movq	%rdx, -48(%rbp)
	jmp	.LBB0_104
	.p2align	4, 0x90
.LBB0_114:
	incq	%rcx
	addq	$10368, %rdx
.LBB0_104:
	testq	%rcx, %rcx
	jg	.LBB0_115
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_106
	.p2align	4, 0x90
.LBB0_113:
	incq	%rdi
	addq	$1296, %rsi
.LBB0_106:
	cmpq	$7, %rdi
	jg	.LBB0_114
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_108
	.p2align	4, 0x90
.LBB0_112:
	incq	%r9
	addq	$72, %r8
.LBB0_108:
	cmpq	$17, %r9
	jg	.LBB0_113
	xorl	%r10d, %r10d
	cmpq	$17, %r10
	jg	.LBB0_112
	.p2align	4, 0x90
.LBB0_111:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$17, %r10
	jle	.LBB0_111
	jmp	.LBB0_112
.LBB0_115:
	movq	%rsp, %r14
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rcx
	movq	%rcx, %rsp
	movl	$14, %r8d
	movq	%r8, -56(%rdx)
	movl	$8, %esi
	movq	%rsi, -64(%rdx)
	movl	$1, %edi
	movq	%rdi, -72(%rdx)
	movq	%rbx, -88(%rdx)
	movq	-72(%rbp), %rsi
	movq	%rsi, -96(%rdx)
	movq	%r8, -48(%rdx)
	movl	$1568, %esi
	movq	%rsi, -40(%rdx)
	movl	$196, %esi
	movq	%rsi, -32(%rdx)
	movq	%r8, -24(%rdx)
	movq	%rdi, -16(%rdx)
	movq	$0, -80(%rdx)
	movq	%rsp, %rdx
	leaq	-96(%rdx), %rdi
	movq	%rdi, %rsp
	movq	-48(%rbp), %rsi
	movq	%rsi, -88(%rdx)
	movq	%rax, -96(%rdx)
	movq	$1, -16(%rdx)
	movq	$18, -24(%rdx)
	movq	$324, -32(%rdx)
	movq	$2592, -40(%rdx)
	movq	$14, -48(%rdx)
	movq	$14, -56(%rdx)
	movq	$8, -64(%rdx)
	movq	$1, -72(%rdx)
	movq	$38, -80(%rdx)
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
	movq	%r14, %rsp
	movl	$12608, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	movq	%rax, -80(%rbp)
	movl	$12608, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ecx, %ecx
	movq	%rax, %rdx
	jmp	.LBB0_116
	.p2align	4, 0x90
.LBB0_126:
	incq	%rcx
	addq	$12544, %rdx
.LBB0_116:
	testq	%rcx, %rcx
	jg	.LBB0_127
	movq	%rdx, %rsi
	xorl	%edi, %edi
	jmp	.LBB0_118
	.p2align	4, 0x90
.LBB0_125:
	incq	%rdi
	addq	$784, %rsi
.LBB0_118:
	cmpq	$15, %rdi
	jg	.LBB0_126
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_120
	.p2align	4, 0x90
.LBB0_124:
	incq	%r9
	addq	$56, %r8
.LBB0_120:
	cmpq	$13, %r9
	jg	.LBB0_125
	xorl	%r10d, %r10d
	cmpq	$13, %r10
	jg	.LBB0_124
	.p2align	4, 0x90
.LBB0_123:
	movl	$0, (%r8,%r10,4)
	incq	%r10
	cmpq	$13, %r10
	jle	.LBB0_123
	jmp	.LBB0_124
.LBB0_127:
	xorl	%ecx, %ecx
	jmp	.LBB0_128
	.p2align	4, 0x90
.LBB0_147:
	incq	%rcx
	addq	$10368, -48(%rbp)
.LBB0_128:
	testq	%rcx, %rcx
	jg	.LBB0_148
	leaq	.L__constant_16x8x5x5xf32(%rip), %rdx
	movq	%rdx, -72(%rbp)
	xorl	%edi, %edi
	jmp	.LBB0_130
	.p2align	4, 0x90
.LBB0_146:
	incq	%rdi
	addq	$800, -72(%rbp)
.LBB0_130:
	cmpq	$15, %rdi
	jg	.LBB0_147
	movq	-48(%rbp), %rdx
	xorl	%ebx, %ebx
	jmp	.LBB0_132
	.p2align	4, 0x90
.LBB0_145:
	incq	%rbx
	movq	-88(%rbp), %rdx
	addq	$72, %rdx
.LBB0_132:
	cmpq	$13, %rbx
	jg	.LBB0_146
	movq	%rdx, -88(%rbp)
	xorl	%r11d, %r11d
	jmp	.LBB0_134
	.p2align	4, 0x90
.LBB0_144:
	incq	%r11
	movq	-56(%rbp), %rdx
	addq	$4, %rdx
.LBB0_134:
	cmpq	$13, %r11
	jg	.LBB0_145
	movq	%rdx, -56(%rbp)
	movq	%rdx, %r14
	movq	-72(%rbp), %rsi
	xorl	%r13d, %r13d
	jmp	.LBB0_136
	.p2align	4, 0x90
.LBB0_143:
	incq	%r13
	addq	$100, %rsi
	movq	-64(%rbp), %r14
	addq	$1296, %r14
.LBB0_136:
	cmpq	$7, %r13
	jg	.LBB0_144
	movq	%r14, -64(%rbp)
	movq	%rsi, %r15
	xorl	%r12d, %r12d
	jmp	.LBB0_138
	.p2align	4, 0x90
.LBB0_142:
	incq	%r12
	addq	$20, %r15
	addq	$72, %r14
.LBB0_138:
	cmpq	$4, %r12
	jg	.LBB0_143
	xorl	%r8d, %r8d
	cmpq	$4, %r8
	jg	.LBB0_142
	.p2align	4, 0x90
.LBB0_141:
	movss	(%r14,%r8,4), %xmm0
	imulq	$3136, %rcx, %rdx
	imulq	$196, %rdi, %r10
	addq	%rdx, %r10
	movq	%rbx, %r9
	shlq	$4, %r9
	subq	%rbx, %r9
	subq	%rbx, %r9
	addq	%r10, %r9
	addq	%r11, %r9
	mulss	(%r15,%r8,4), %xmm0
	addss	(%rax,%r9,4), %xmm0
	movss	%xmm0, (%rax,%r9,4)
	incq	%r8
	cmpq	$4, %r8
	jle	.LBB0_141
	jmp	.LBB0_142
.LBB0_148:
	xorl	%ecx, %ecx
	leaq	.L__constant_16x1x1xf32(%rip), %rdx
	movq	-80(%rbp), %rsi
	jmp	.LBB0_149
	.p2align	4, 0x90
.LBB0_159:
	incq	%rcx
	addq	$12544, %rsi
.LBB0_149:
	testq	%rcx, %rcx
	jg	.LBB0_160
	movq	%rax, %rdi
	movq	%rsi, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_151
	.p2align	4, 0x90
.LBB0_158:
	incq	%r9
	addq	$784, %r8
	addq	$784, %rdi
.LBB0_151:
	cmpq	$15, %r9
	jg	.LBB0_159
	movq	%rdi, %r10
	movq	%r8, %r11
	xorl	%ebx, %ebx
	jmp	.LBB0_153
	.p2align	4, 0x90
.LBB0_157:
	incq	%rbx
	addq	$56, %r11
	addq	$56, %r10
.LBB0_153:
	cmpq	$13, %rbx
	jg	.LBB0_158
	xorl	%r14d, %r14d
	cmpq	$13, %r14
	jg	.LBB0_157
	.p2align	4, 0x90
.LBB0_156:
	movss	(%r10,%r14,4), %xmm0
	addss	(%rdx,%r9,4), %xmm0
	movss	%xmm0, (%r11,%r14,4)
	incq	%r14
	cmpq	$13, %r14
	jle	.LBB0_156
	jmp	.LBB0_157
.LBB0_160:
	xorl	%eax, %eax
	xorps	%xmm0, %xmm0
	movq	-80(%rbp), %rcx
	jmp	.LBB0_161
	.p2align	4, 0x90
.LBB0_171:
	incq	%rax
	addq	$12544, %rcx
.LBB0_161:
	testq	%rax, %rax
	jg	.LBB0_172
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_163
	.p2align	4, 0x90
.LBB0_170:
	incq	%rsi
	addq	$784, %rdx
.LBB0_163:
	cmpq	$15, %rsi
	jg	.LBB0_171
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_165
	.p2align	4, 0x90
.LBB0_169:
	incq	%r8
	addq	$56, %rdi
.LBB0_165:
	cmpq	$13, %r8
	jg	.LBB0_170
	xorl	%r9d, %r9d
	cmpq	$13, %r9
	jg	.LBB0_169
	.p2align	4, 0x90
.LBB0_168:
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
	jle	.LBB0_168
	jmp	.LBB0_169
.LBB0_172:
	movl	$1088, %edi
	callq	malloc@PLT
	movq	%rax, %r12
	addq	$63, %r12
	andq	$-64, %r12
	xorl	%eax, %eax
	movq	%r12, %rcx
	jmp	.LBB0_173
	.p2align	4, 0x90
.LBB0_183:
	incq	%rax
	addq	$1024, %rcx
.LBB0_173:
	testq	%rax, %rax
	jg	.LBB0_184
	movq	%rcx, %rdx
	xorl	%esi, %esi
	jmp	.LBB0_175
	.p2align	4, 0x90
.LBB0_182:
	incq	%rsi
	addq	$64, %rdx
.LBB0_175:
	cmpq	$15, %rsi
	jg	.LBB0_183
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_177
	.p2align	4, 0x90
.LBB0_181:
	incq	%r8
	addq	$16, %rdi
.LBB0_177:
	cmpq	$3, %r8
	jg	.LBB0_182
	xorl	%r9d, %r9d
	cmpq	$3, %r9
	jg	.LBB0_181
	.p2align	4, 0x90
.LBB0_180:
	movl	$-8388608, (%rdi,%r9,4)
	incq	%r9
	cmpq	$3, %r9
	jle	.LBB0_180
	jmp	.LBB0_181
.LBB0_184:
	movl	$1088, %edi
	callq	malloc@PLT
	movq	%rax, %r15
	addq	$63, %r15
	andq	$-64, %r15
	movl	$1024, %edx
	movq	%r15, %rdi
	movq	%r12, %rsi
	callq	memcpy@PLT
	xorl	%eax, %eax
	movq	-80(%rbp), %rdx
	jmp	.LBB0_185
	.p2align	4, 0x90
.LBB0_205:
	movq	-56(%rbp), %rax
	incq	%rax
	movq	-80(%rbp), %rdx
	addq	$12544, %rdx
.LBB0_185:
	testq	%rax, %rax
	jg	.LBB0_206
	movq	%rax, -56(%rbp)
	shlq	$8, %rax
	movq	%rax, -64(%rbp)
	movq	%rdx, -80(%rbp)
	xorl	%esi, %esi
	jmp	.LBB0_187
	.p2align	4, 0x90
.LBB0_204:
	incq	%rsi
	addq	$784, %rdx
.LBB0_187:
	cmpq	$15, %rsi
	jg	.LBB0_205
	movq	%rsi, %rdi
	shlq	$4, %rdi
	addq	-64(%rbp), %rdi
	movq	%rdx, %r8
	xorl	%r9d, %r9d
	jmp	.LBB0_189
	.p2align	4, 0x90
.LBB0_203:
	incq	%r9
	addq	$168, %r8
.LBB0_189:
	cmpq	$3, %r9
	jg	.LBB0_204
	leaq	(%rdi,%r9,4), %r10
	movq	%r8, %r11
	xorl	%ebx, %ebx
	jmp	.LBB0_191
	.p2align	4, 0x90
.LBB0_202:
	incq	%rbx
	addq	$12, %r11
.LBB0_191:
	cmpq	$3, %rbx
	jg	.LBB0_203
	leaq	(%r10,%rbx), %r12
	movq	%r11, %r13
	xorl	%r14d, %r14d
	jmp	.LBB0_193
	.p2align	4, 0x90
.LBB0_201:
	incq	%r14
	addq	$56, %r13
.LBB0_193:
	cmpq	$2, %r14
	jg	.LBB0_202
	xorl	%eax, %eax
	jmp	.LBB0_195
	.p2align	4, 0x90
.LBB0_200:
	movdqa	%xmm1, %xmm2
	cmpunordss	%xmm1, %xmm2
	movaps	%xmm2, %xmm3
	andps	%xmm1, %xmm3
	maxss	%xmm0, %xmm1
	andnps	%xmm1, %xmm2
	orps	%xmm3, %xmm2
	movss	%xmm2, (%r15,%r12,4)
	incq	%rax
.LBB0_195:
	cmpq	$2, %rax
	jg	.LBB0_201
	movd	(%r13,%rax,4), %xmm0
	movd	(%r15,%r12,4), %xmm2
	movd	%xmm2, %ecx
	testl	%ecx, %ecx
	movdqa	%xmm2, %xmm1
	js	.LBB0_198
	movdqa	%xmm0, %xmm1
.LBB0_198:
	js	.LBB0_200
	movdqa	%xmm2, %xmm0
	jmp	.LBB0_200
.LBB0_206:
	movl	$104, %edi
	callq	malloc@PLT
	addq	$63, %rax
	andq	$-64, %rax
	xorl	%ecx, %ecx
	movq	%rax, %rdx
	jmp	.LBB0_207
	.p2align	4, 0x90
.LBB0_211:
	incq	%rcx
	addq	$40, %rdx
.LBB0_207:
	testq	%rcx, %rcx
	jg	.LBB0_212
	xorl	%esi, %esi
	cmpq	$9, %rsi
	jg	.LBB0_211
	.p2align	4, 0x90
.LBB0_210:
	movl	$0, (%rdx,%rsi,4)
	incq	%rsi
	cmpq	$9, %rsi
	jle	.LBB0_210
	jmp	.LBB0_211
.LBB0_212:
	xorl	%ecx, %ecx
	leaq	.L__constant_16x4x4x10xf32(%rip), %rdx
	jmp	.LBB0_213
	.p2align	4, 0x90
.LBB0_220:
	incq	%rcx
	addq	$1024, %r15
.LBB0_213:
	testq	%rcx, %rcx
	jg	.LBB0_221
	leaq	(%rcx,%rcx,4), %rsi
	movq	%rdx, %rdi
	xorl	%r8d, %r8d
	jmp	.LBB0_215
	.p2align	4, 0x90
.LBB0_219:
	incq	%r8
	addq	$4, %rdi
.LBB0_215:
	cmpq	$9, %r8
	jg	.LBB0_220
	leaq	(%r8,%rsi,2), %r9
	movq	%rdi, %r10
	xorl	%r11d, %r11d
	cmpq	$255, %r11
	jg	.LBB0_219
	.p2align	4, 0x90
.LBB0_218:
	movss	(%r15,%r11,4), %xmm0
	mulss	(%r10), %xmm0
	addss	(%rax,%r9,4), %xmm0
	movss	%xmm0, (%rax,%r9,4)
	incq	%r11
	addq	$40, %r10
	cmpq	$255, %r11
	jle	.LBB0_218
	jmp	.LBB0_219
.LBB0_221:
	leaq	.L__constant_1x10xf32(%rip), %rcx
	xorl	%edx, %edx
	movq	%rax, %rsi
	jmp	.LBB0_222
	.p2align	4, 0x90
.LBB0_226:
	incq	%rdx
	addq	$40, %rcx
	addq	$40, %rsi
.LBB0_222:
	testq	%rdx, %rdx
	jg	.LBB0_227
	xorl	%edi, %edi
	cmpq	$9, %rdi
	jg	.LBB0_226
	.p2align	4, 0x90
.LBB0_225:
	movss	(%rsi,%rdi,4), %xmm0
	addss	(%rcx,%rdi,4), %xmm0
	movss	%xmm0, (%rsi,%rdi,4)
	incq	%rdi
	cmpq	$9, %rdi
	jle	.LBB0_225
	jmp	.LBB0_226
.LBB0_227:
	movq	32(%rax), %rcx
	movq	-96(%rbp), %rdx
	movq	%rcx, 32(%rdx)
	movq	24(%rax), %rcx
	movq	%rcx, 24(%rdx)
	movq	16(%rax), %rcx
	movq	%rcx, 16(%rdx)
	movq	(%rax), %rcx
	movq	8(%rax), %rax
	movq	%rax, 8(%rdx)
	movq	%rcx, (%rdx)
	xorl	%eax, %eax
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

	.type	.L__constant_16x8x5x5xf32,@object
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
.L__constant_16x8x5x5xf32:
	.long	0xbd46e30a
	.long	0xbdba655c
	.long	0xbbd737af
	.long	0xbe037c31
	.long	0x3ddf0715
	.long	0x3dac86cf
	.long	0xbe2c6e48
	.long	0xbe8aeb83
	.long	0xbc9bdb4d
	.long	0x3d8dd988
	.long	0x3d22d19c
	.long	0xbc40ee1a
	.long	0xbd816931
	.long	0xbe1a7b07
	.long	0xbde648fb
	.long	0x3d8f193f
	.long	0x3a498789
	.long	0x3daab789
	.long	0xbe396445
	.long	0xbdf2b32a
	.long	0x3b5d8e45
	.long	0x3e80aa81
	.long	0x3e464319
	.long	0xbd5f53f7
	.long	0xbe033735
	.long	0xbed91cdd
	.long	0xbe8a8f46
	.long	0xbe84d68c
	.long	0x3da5f721
	.long	0x3db429ed
	.long	0x3c1f008f
	.long	0xbe6aef3c
	.long	0xbd831b23
	.long	0x3cb2bb78
	.long	0xbd4b47a5
	.long	0x3e2d88a6
	.long	0x3e714404
	.long	0xbc9f35fb
	.long	0xbe0a4619
	.long	0xbd2f5844
	.long	0xbd87423e
	.long	0x3e3e9b02
	.long	0x3e39d755
	.long	0xbe145cf9
	.long	0x3cf69cf6
	.long	0xbe0f343a
	.long	0x3de8a96f
	.long	0x3dd7236d
	.long	0x3d5218b2
	.long	0x3d6cb203
	.long	0xbe1ec967
	.long	0xbe0931df
	.long	0x3da182ec
	.long	0x3dee9550
	.long	0xbe0a8d72
	.long	0xbdff049b
	.long	0xbe541cf2
	.long	0xbe11d08d
	.long	0xbd7bb9fc
	.long	0xbdadb390
	.long	0xbe41fb75
	.long	0xbe61ec6a
	.long	0xbe5dc3cb
	.long	0xbdb18718
	.long	0x3d84eda4
	.long	0xbe84c2ad
	.long	0xbdd4db8b
	.long	0xbeaa7b09
	.long	0xbe09e19f
	.long	0xbd019f2b
	.long	0x3ce7b278
	.long	0x3db76bed
	.long	0xbddea708
	.long	0xbe3ff000
	.long	0xbe652e01
	.long	0x3dd351d8
	.long	0xbd42fe67
	.long	0xbcd91419
	.long	0xbcb0143c
	.long	0x3ceb1d4e
	.long	0x3d440761
	.long	0xbacb517f
	.long	0x3dd4b53e
	.long	0xbdb44c46
	.long	0xbd60fcbe
	.long	0x3d9d1b8c
	.long	0x3cd86732
	.long	0xbe83fb03
	.long	0xbe91abe6
	.long	0xbe55a122
	.long	0x3c9290af
	.long	0x3d8add1e
	.long	0xbd0cb83c
	.long	0xbe9bf4fa
	.long	0xbe01aced
	.long	0xbc2ace68
	.long	0x3df07a84
	.long	0x3e2df710
	.long	0xbd895728
	.long	0x3d617e19
	.long	0x3e3308fe
	.long	0xbd862c04
	.long	0xbe0e516d
	.long	0xbe9e02cd
	.long	0xbdd605f7
	.long	0x3dbd790a
	.long	0x3e999422
	.long	0x3d7a66d6
	.long	0xbe65438f
	.long	0xbe5ef7d9
	.long	0xbdb1973c
	.long	0x3d0d7883
	.long	0x3df982a0
	.long	0xbe12dcd9
	.long	0xbe10bab3
	.long	0xbed8b560
	.long	0xbd110363
	.long	0x3d59cabd
	.long	0x3cc62768
	.long	0xbc83a796
	.long	0xbdf6991e
	.long	0xbe268155
	.long	0xbdc6229f
	.long	0x3d12065f
	.long	0x3c5d92bd
	.long	0x3e134ecc
	.long	0x3ca17c5f
	.long	0xbeaca26f
	.long	0xbe6482ff
	.long	0xbe25eb31
	.long	0x3e2b7f98
	.long	0x3ee770df
	.long	0x3e891ff8
	.long	0xbdffadbe
	.long	0xbe5f7e19
	.long	0xbe18bdcd
	.long	0xbd9bbbe2
	.long	0x3e83d750
	.long	0x3dd2b0c3
	.long	0x3d854f7f
	.long	0xbed78292
	.long	0xbe9f7469
	.long	0xbd1384cf
	.long	0x3d3fab07
	.long	0x3dea8549
	.long	0xbd5a0f39
	.long	0xbe0b6d56
	.long	0xbe885dfd
	.long	0xbe5a2e5a
	.long	0xbcfdbfef
	.long	0x3d5e9adb
	.long	0xbe2e9ecf
	.long	0xbe945b2f
	.long	0xbd29a79f
	.long	0xbe7c8f35
	.long	0x3dcb5f0d
	.long	0x3e47ee52
	.long	0xbdc90d18
	.long	0xbe8a7d92
	.long	0xbe195d17
	.long	0xbdaf41ba
	.long	0x3b01952f
	.long	0x3d5d8888
	.long	0x3d3f7a7a
	.long	0xbe24f1c1
	.long	0x3cbeb4da
	.long	0x3d859159
	.long	0x3e7bd48b
	.long	0x3d786041
	.long	0x3c523a70
	.long	0xbdc1f94b
	.long	0xbdd8b429
	.long	0x3e6ad8c9
	.long	0x3e5134ef
	.long	0xbdb81bd3
	.long	0xbe7d7b29
	.long	0x3e13bb95
	.long	0x3d54d24f
	.long	0xbcc44019
	.long	0x3dcccd5f
	.long	0xbd69f888
	.long	0xbd9b58be
	.long	0xbd6b4c52
	.long	0x3d51903c
	.long	0x3db152f4
	.long	0x3d7f14e4
	.long	0xbe1e351c
	.long	0xbe0fe39b
	.long	0x3ca59e74
	.long	0x3ce2691a
	.long	0xbdd02324
	.long	0x3ceb5e80
	.long	0xbe0314d4
	.long	0xbe687294
	.long	0xbe899d6f
	.long	0x3e08704e
	.long	0x3d9fe2ff
	.long	0xbdfcd4fb
	.long	0xbe14e319
	.long	0xbe310e12
	.long	0x3d9969cc
	.long	0x3e5be042
	.long	0xbc98dac4
	.long	0xbc3e581a
	.long	0xbe218d53
	.long	0xbd20f2d4
	.long	0xbc6af610
	.long	0xbd0f21a8
	.long	0x3e0cd008
	.long	0x3e80e502
	.long	0x3d332743
	.long	0xbec90b86
	.long	0xbd9f8de0
	.long	0x3e199365
	.long	0x3e532147
	.long	0xbe10506c
	.long	0xbe45c78c
	.long	0xbd319a83
	.long	0x3db8c873
	.long	0x3e5e4847
	.long	0xbdac090f
	.long	0xbe4f14df
	.long	0xbe06b99e
	.long	0xbdcc8a83
	.long	0xbae1472e
	.long	0xbebc27f1
	.long	0xbe9ec786
	.long	0xbed32f63
	.long	0xbd04aff5
	.long	0xbe13bfba
	.long	0xbdaa6fdd
	.long	0xbd84ee9f
	.long	0xbe8822b8
	.long	0xbd71e85f
	.long	0xbdd18b55
	.long	0x3d14ed38
	.long	0x3e0c6ab7
	.long	0xbd1987c2
	.long	0xbdbf2242
	.long	0x3be215be
	.long	0x3ca5da97
	.long	0x3e6b6e88
	.long	0xbc13bfd4
	.long	0xbe2f48d3
	.long	0xbe38bc28
	.long	0x3cbdcc12
	.long	0xbe1dbc3c
	.long	0xbc561844
	.long	0xbd07fdce
	.long	0xbdae84cf
	.long	0xbe633fac
	.long	0xbe811778
	.long	0xbe3d28c1
	.long	0xbd664be3
	.long	0x3dc399e7
	.long	0xbdd458ce
	.long	0xbe2ddb33
	.long	0xbe21239b
	.long	0x3aeed93d
	.long	0x3bc8463f
	.long	0xbbc493f2
	.long	0x3e560ed1
	.long	0x3d82dd6d
	.long	0x3e295c14
	.long	0x3e8da248
	.long	0xbd6b7d66
	.long	0x3db23665
	.long	0x3e174693
	.long	0x3e0bba3d
	.long	0xbc1c4823
	.long	0x3e02e2cb
	.long	0xbc90cfd9
	.long	0xbd9f87e2
	.long	0xbd05fa57
	.long	0xbe1b44ac
	.long	0x3e0baf92
	.long	0x3cf64619
	.long	0xbd926298
	.long	0xbe212377
	.long	0xbda38f3e
	.long	0x3cc7239a
	.long	0x3bd79bd6
	.long	0xbe0e001f
	.long	0x3de85037
	.long	0x3e732417
	.long	0xbd895924
	.long	0xbe0250b4
	.long	0xbcc67912
	.long	0x3e5ee4bb
	.long	0x3e764ab3
	.long	0xbe69aeb6
	.long	0xbc0c7409
	.long	0xbe48e323
	.long	0xbdb6e4c7
	.long	0xbe36b739
	.long	0xbe4be899
	.long	0xbe9e7275
	.long	0xbde2b386
	.long	0xbdf46ac6
	.long	0x3d2ac405
	.long	0x3c3c1a2b
	.long	0xbdb987e5
	.long	0xbe255c85
	.long	0xbe257e86
	.long	0xbdbfc179
	.long	0x3dcf44d5
	.long	0x3e6c18e8
	.long	0x3eab1023
	.long	0xbe2289ac
	.long	0xbce54669
	.long	0x3e1beed5
	.long	0x3d07d4bb
	.long	0x3e50f70e
	.long	0x3ddc5978
	.long	0x3e02389c
	.long	0x3cd88a84
	.long	0xbe3df94e
	.long	0xbe5c5c39
	.long	0xbe1b1287
	.long	0xbe470829
	.long	0xbdd03c41
	.long	0xbd88da2a
	.long	0xbe39f3db
	.long	0xbe66dfdd
	.long	0xbd7edeab
	.long	0xbe576570
	.long	0xbdb27139
	.long	0xbe06d87a
	.long	0xbdf3f0ca
	.long	0xbe192b5f
	.long	0x3dccadfe
	.long	0x3df4d4d0
	.long	0x3e69ec5c
	.long	0xbcea2ebb
	.long	0x3d3524ff
	.long	0x3e214c40
	.long	0x3dbc2d6c
	.long	0x3e68e700
	.long	0xbcb2a5ab
	.long	0x3e4449f7
	.long	0x3d96eb3a
	.long	0x3d486695
	.long	0xbd2099dc
	.long	0xbd40ba91
	.long	0xbe6a6900
	.long	0xbd140855
	.long	0x3d29aeab
	.long	0xbd9ef94c
	.long	0xbe85b1e7
	.long	0xbdbac77c
	.long	0xbe3113b9
	.long	0xbe74103b
	.long	0x3dd5aed5
	.long	0xbcff7560
	.long	0xbb82ec29
	.long	0xbd4c5586
	.long	0x3e0bbaff
	.long	0x3e5e73e0
	.long	0xbd4d7d56
	.long	0xbd2bf0d1
	.long	0xbd81eb4d
	.long	0xbd8d218f
	.long	0x3c0befd1
	.long	0x3e797405
	.long	0x3e0d3a2f
	.long	0xbc36bdeb
	.long	0xbe125c0b
	.long	0x3de460cb
	.long	0x3d95c7cf
	.long	0xbdb3ca24
	.long	0x3d492d7d
	.long	0xbd3f8d1b
	.long	0xbe4009f1
	.long	0xbe2a63fc
	.long	0xbd5010e3
	.long	0xbdad94e2
	.long	0x3cdbc28d
	.long	0xbdb1232a
	.long	0xbe3ef48c
	.long	0x3bf13768
	.long	0xbe0ba76f
	.long	0xbe76ea1f
	.long	0xbe8d13bd
	.long	0xbdd91c8c
	.long	0xba5729a9
	.long	0xbd9e8d3d
	.long	0x3de69da7
	.long	0x3dc8e7da
	.long	0xbd224687
	.long	0x3e032122
	.long	0x3db33ede
	.long	0x3e8a7497
	.long	0x3e4f8c7a
	.long	0x3e548064
	.long	0x3d2304f4
	.long	0xbe18dcdc
	.long	0xbdbfa804
	.long	0xbddce173
	.long	0xbe476268
	.long	0xbe8aa66d
	.long	0xbe207050
	.long	0xbcda15d2
	.long	0x3dc6b84c
	.long	0xbc0e5a96
	.long	0xbc775261
	.long	0xbe37f2d2
	.long	0xbd700f1c
	.long	0xbda2036c
	.long	0xbe61f988
	.long	0x3ddc5985
	.long	0xbf024483
	.long	0xbe9eac74
	.long	0xbe11d673
	.long	0x3c6407d8
	.long	0xbd084862
	.long	0xbd7a9084
	.long	0xbe6972fb
	.long	0xbe2cf533
	.long	0xbdf06547
	.long	0xbdba5a37
	.long	0x3d8506f0
	.long	0xbe7146ac
	.long	0xbc8936cc
	.long	0x3cef60ac
	.long	0x3e56b567
	.long	0xbe2f732c
	.long	0x3ea0b422
	.long	0x3e6368a7
	.long	0xbe7319d5
	.long	0xbe2ede08
	.long	0xbe5f3a8c
	.long	0x3e4b41e0
	.long	0x3e2cfb0b
	.long	0x3d7c2d4e
	.long	0xbd2f5b15
	.long	0xbe8969c0
	.long	0x3d87cc92
	.long	0x3e9575a1
	.long	0x3d19ccd2
	.long	0x3e2b599b
	.long	0xbea5cf63
	.long	0xbe848a57
	.long	0xbd94b75a
	.long	0xbe48139e
	.long	0x3d911b76
	.long	0x3c9d6791
	.long	0xbd957548
	.long	0xbd616d8a
	.long	0xbcd84b90
	.long	0xbe408046
	.long	0x3dbe402a
	.long	0xbc41d69e
	.long	0xbe3cace9
	.long	0xbcf8e83d
	.long	0xbda6048c
	.long	0x3df24bd4
	.long	0x3d0b13b9
	.long	0xbe1b3bf5
	.long	0xbdff3965
	.long	0x3eb84110
	.long	0x3e9b9295
	.long	0x3e239ce7
	.long	0x3df07996
	.long	0x3e32038b
	.long	0x3e5e7429
	.long	0xbcc9b546
	.long	0xbd547d90
	.long	0xbca4abd0
	.long	0x3dd9ff50
	.long	0xbdec76fa
	.long	0x3dd1f00e
	.long	0xbd82162e
	.long	0xbdee9b3d
	.long	0xbe08bc0d
	.long	0xbd94eb07
	.long	0x3da01c58
	.long	0x3ca105ab
	.long	0x3dd97c2e
	.long	0x3da954c0
	.long	0x3e1e130b
	.long	0xbe97119b
	.long	0xbe44ba9d
	.long	0xbd95ae50
	.long	0x3d43c51f
	.long	0x3e4ead17
	.long	0xbe62d247
	.long	0xbe92554e
	.long	0xbec9c952
	.long	0xbe04317d
	.long	0xbd8c592b
	.long	0x3e822e2b
	.long	0x3dc78fd3
	.long	0xbd6f9655
	.long	0xbe0c9fda
	.long	0xbe8183e9
	.long	0x3e87daea
	.long	0x3d39e5db
	.long	0x3d7ef8ea
	.long	0x3e37dd23
	.long	0x3e337710
	.long	0x3db54e36
	.long	0x3cbd58c0
	.long	0xbd9dbdc8
	.long	0x3d9feef9
	.long	0xbd3eb9cf
	.long	0xbdac8c00
	.long	0xbd84575a
	.long	0xbe00884f
	.long	0xbdac22f6
	.long	0x3c68b840
	.long	0x3d110e7c
	.long	0xbe2897e4
	.long	0xbe7a80da
	.long	0xbebd233b
	.long	0xbe38d5c7
	.long	0x3e0ebbf3
	.long	0x3dd2c185
	.long	0xbdccbf3d
	.long	0xbe1f177c
	.long	0xbecd8ebf
	.long	0x3da414b1
	.long	0x3dee5067
	.long	0x3d47fcde
	.long	0x3e53e20c
	.long	0x3dd67e90
	.long	0xbb980f73
	.long	0x3b772a81
	.long	0xbe415a28
	.long	0x3dbb7742
	.long	0xbdb5a382
	.long	0xbd3e3e09
	.long	0xbe259da4
	.long	0xbd1d4ed2
	.long	0xbd4733aa
	.long	0xbd548813
	.long	0x3d86fc74
	.long	0x3cb3cc0f
	.long	0xbe02a37c
	.long	0xbe464934
	.long	0xbe8a48ad
	.long	0x3dc14c5e
	.long	0x3e68c3aa
	.long	0x3dc34cd8
	.long	0xbd26cd0b
	.long	0xbe73a321
	.long	0xbe0f6b3f
	.long	0x3d243a69
	.long	0x3e5bf1f8
	.long	0x3d8e2fe8
	.long	0x3e9bfc2e
	.long	0xbd8a6bdf
	.long	0xbd772396
	.long	0xbc636a44
	.long	0x3e64a794
	.long	0x3df6a963
	.long	0xbd8003cd
	.long	0xbd9f31ca
	.long	0x3b8fd18b
	.long	0xbd8742bd
	.long	0xbe9a0802
	.long	0xbe0e101a
	.long	0xbe62361c
	.long	0xbe25325f
	.long	0xbe8b5972
	.long	0xbe9164d1
	.long	0xbe5cceb9
	.long	0xbd970a3d
	.long	0xbe48a6ce
	.long	0xbe8be500
	.long	0xbe866831
	.long	0xbe3ec456
	.long	0x3e66dbbe
	.long	0x3cad8424
	.long	0x3a54ec4a
	.long	0x3c37451e
	.long	0x3d86ce8c
	.long	0xbd09f953
	.long	0xbe537614
	.long	0xbe528b14
	.long	0xbe547104
	.long	0xbca27642
	.long	0x3d4ee0c8
	.long	0x3b9a8a0e
	.long	0xbd4e14aa
	.long	0x3e71e2be
	.long	0xbd83f3a0
	.long	0x3d5b0303
	.long	0x3e89be62
	.long	0x3eb0f6e5
	.long	0x3e5164f9
	.long	0xbed4bd6d
	.long	0xbe7c0d21
	.long	0xbdd5ff0b
	.long	0xbd378714
	.long	0x3cdca0c2
	.long	0x3db250b6
	.long	0xbe65cc77
	.long	0xbe6a7e06
	.long	0x3bd5439d
	.long	0xbc941708
	.long	0x3eac38f5
	.long	0x3d533b21
	.long	0xbdd1a3b3
	.long	0xbd6cd68c
	.long	0xbe2419fe
	.long	0xbc1ee893
	.long	0xbe563f90
	.long	0x3db2ecbc
	.long	0xbdca43ac
	.long	0xbe0bcf53
	.long	0xbd5049e5
	.long	0xbe234f1a
	.long	0x3da66661
	.long	0x3d6ff228
	.long	0xbd67af1a
	.long	0xbc3db224
	.long	0xbe00bc0e
	.long	0x3e1d5d17
	.long	0xbd10eca7
	.long	0xbca780ec
	.long	0xbe29a983
	.long	0xbea973c5
	.long	0xbe65989b
	.long	0x3c950e29
	.long	0xbe69639f
	.long	0x3d8e01bc
	.long	0x3dff8c98
	.long	0x3dfe5acb
	.long	0xbd4d9eac
	.long	0x3abe1d55
	.long	0x3dc7c068
	.long	0x3e85e1cb
	.long	0x3d2df667
	.long	0xbdbbed1c
	.long	0x3e883635
	.long	0x3c455440
	.long	0x3d820731
	.long	0x3e955df9
	.long	0xbe56d86b
	.long	0xbda5db28
	.long	0xbe1b18f3
	.long	0x3c8abbd4
	.long	0x3cd70789
	.long	0xbe93501d
	.long	0xbe1f5e05
	.long	0xbded52f0
	.long	0xbe29ec3e
	.long	0xbd4715d1
	.long	0xbdbb502f
	.long	0xbeaf51e9
	.long	0xbda8ee80
	.long	0xbe15e9c3
	.long	0x3d7cbf99
	.long	0xbd8392a8
	.long	0x3dd7b1ab
	.long	0x3d058429
	.long	0xbd2476ae
	.long	0xbd2b6ada
	.long	0xbe15edc0
	.long	0x3d3e77bd
	.long	0xbe4430bc
	.long	0xbdc71226
	.long	0xbcdb3e99
	.long	0xbc0eb6bf
	.long	0xbe3bd884
	.long	0x3d0510da
	.long	0x3d60da87
	.long	0x3d677535
	.long	0x3db95372
	.long	0xbd952a87
	.long	0x3d26ffb6
	.long	0x3c9107f8
	.long	0x3cbdc5c1
	.long	0x3daeec18
	.long	0x3d961068
	.long	0x3d974f4d
	.long	0x3e4987b5
	.long	0x3e0811ef
	.long	0xbda61c98
	.long	0xbdbc8371
	.long	0xbcad8d1c
	.long	0x3d850882
	.long	0x3c886029
	.long	0xbe4cb86e
	.long	0xbec8b6a0
	.long	0x3d5aa3c1
	.long	0x3e17e15c
	.long	0xbc8280af
	.long	0xbd3fc7d2
	.long	0xbe402d36
	.long	0xbd1ed832
	.long	0xbdc4c427
	.long	0xbe773469
	.long	0xbde1fdde
	.long	0xbdffa30e
	.long	0xbe46d71b
	.long	0xbe5833d8
	.long	0xbdb701b8
	.long	0xbde5b91d
	.long	0xbe4a83ef
	.long	0xbde9b5e3
	.long	0xbd741a63
	.long	0x3dca2d85
	.long	0xbb47b052
	.long	0x3dc23fd6
	.long	0xbe8d669f
	.long	0xbe8d7084
	.long	0x3ddbf19d
	.long	0x3e2ae9ab
	.long	0xbe1f080b
	.long	0xbdcdc89a
	.long	0xbd671569
	.long	0xbe2d75b1
	.long	0x3de28500
	.long	0xbe03c3d0
	.long	0xbe691e1c
	.long	0xbdc53c7c
	.long	0xbdd0bd78
	.long	0xbdd1f4d1
	.long	0xbdba6f2b
	.long	0xbdca50a6
	.long	0xbe01a8c5
	.long	0xbd8fa7f7
	.long	0xbd0987cf
	.long	0xbe50da91
	.long	0xbe0ea5ad
	.long	0xbe662a6f
	.long	0xbdab20c4
	.long	0x3cc37472
	.long	0x3e24ba7b
	.long	0xbe21a8f8
	.long	0xbe7213b8
	.long	0xbe3f1368
	.long	0x3da9dec7
	.long	0x3bfce1c4
	.long	0xbe427707
	.long	0xbd16dcb2
	.long	0xbe119238
	.long	0xbd3bebaa
	.long	0x3d1743a0
	.long	0xbe90dfc9
	.long	0xbe297d5c
	.long	0xbe3df67e
	.long	0xbe6b9154
	.long	0xbd368eab
	.long	0xbe5429c7
	.long	0xbd7f1e19
	.long	0xbcd9c4cb
	.long	0xbdb9f6a3
	.long	0xbc90ecb3
	.long	0xbe5ae904
	.long	0xbe2e0781
	.long	0xbd4eea36
	.long	0x3e247486
	.long	0xbdd67a63
	.long	0xbe3cbc9d
	.long	0xbda76e59
	.long	0x3cc59f96
	.long	0x3ea1e9d9
	.long	0xbca8ec1d
	.long	0xbd17cb6d
	.long	0xbc823c57
	.long	0x3d8edca7
	.long	0x3ed0e981
	.long	0x3e44d146
	.long	0xbe1bbac5
	.long	0xbda4f8b2
	.long	0x3e427e05
	.long	0x3e82dbab
	.long	0x3bb9a8d9
	.long	0xbd823986
	.long	0xbcc0a020
	.long	0x3ce9bda2
	.long	0xbddd26df
	.long	0xbda08a03
	.long	0x3be2e7b9
	.long	0xbcf57654
	.long	0x3e50804c
	.long	0x3dd3010a
	.long	0x3d7b5fe7
	.long	0xbdd8eb3c
	.long	0x3d9d14ce
	.long	0x3a88e92d
	.long	0x3e44d237
	.long	0x3d6f5239
	.long	0xbe562d9e
	.long	0xbde2c9ac
	.long	0x3e1b4e48
	.long	0xbd181c58
	.long	0xbd5de7c2
	.long	0xbd66834b
	.long	0x3e2d3b3a
	.long	0x3e310890
	.long	0xbd932de1
	.long	0xbe1e8c0a
	.long	0x3dc74572
	.long	0xbdb0ade9
	.long	0x3d1aa41a
	.long	0xbdf8b645
	.long	0xbe34a397
	.long	0xbd195ecf
	.long	0xbdf82bd1
	.long	0xbd854d0b
	.long	0xbc3defbb
	.long	0xbe3e72dc
	.long	0x3e33c55d
	.long	0x3e698886
	.long	0x3de66d49
	.long	0x3dab0306
	.long	0xbdf6d06c
	.long	0x3e2ff3d5
	.long	0x3e5aa3de
	.long	0x3d603f02
	.long	0x3e8355cf
	.long	0xbd38e5f9
	.long	0xbd9d20a2
	.long	0xbcf1037c
	.long	0x3ec30825
	.long	0x3df03e8a
	.long	0x3c81bb9a
	.long	0xbe379413
	.long	0xbc90224b
	.long	0x3dd70e79
	.long	0x3be750e9
	.long	0xbd754ab0
	.long	0xbb929ba5
	.long	0xbe0fcb52
	.long	0xbd88cbb5
	.long	0xbe472fbb
	.long	0xbd444f64
	.long	0x3c66b911
	.long	0xbd598c56
	.long	0x3dc7941f
	.long	0x3e3aba5f
	.long	0x3e8b75ca
	.long	0x3e17d536
	.long	0xbddf2fcb
	.long	0xbbc91fd8
	.long	0x3c6e3f60
	.long	0x3cef1507
	.long	0xbd2dac00
	.long	0xbe15339e
	.long	0xbe87bc9b
	.long	0xbe4f7302
	.long	0xbe6d65c2
	.long	0x3e440e22
	.long	0x3e08b906
	.long	0xbe1bb323
	.long	0xbe619c86
	.long	0xbe9652e2
	.long	0xbd93d984
	.long	0xbd566ee7
	.long	0x3d905043
	.long	0x3e2b1a6e
	.long	0x3e26ef39
	.long	0xbd39b676
	.long	0xbe2956af
	.long	0xbe21353c
	.long	0xbdb09b3b
	.long	0xbc0f96d7
	.long	0xbd85c179
	.long	0xbdefd74d
	.long	0xbe940376
	.long	0xbe11c33b
	.long	0xbd8f3ba7
	.long	0xbd964eb4
	.long	0x3d4099ce
	.long	0x3d90c465
	.long	0x3c3ec091
	.long	0x3ac88ba5
	.long	0x3e892fc0
	.long	0x3ed45b93
	.long	0x3ea9ecbb
	.long	0x3d59761d
	.long	0xbdbae9ef
	.long	0xbdba7d88
	.long	0xbda4eb71
	.long	0xbd9ed156
	.long	0xbe1bd59e
	.long	0xbd8e382a
	.long	0xbe238dfa
	.long	0x3c901b17
	.long	0x3c896b5e
	.long	0xbdd583b8
	.long	0xbe1a6acb
	.long	0xbe0aad00
	.long	0xbc6114d8
	.long	0xbcdf2b0e
	.long	0x3df8f7da
	.long	0xbe4afec8
	.long	0xbcd55bf0
	.long	0x3cd20586
	.long	0xbd71cdfd
	.long	0x3b66ffa0
	.long	0xbdc14ed2
	.long	0xbe05d55a
	.long	0xbe71a484
	.long	0xbddce0a7
	.long	0xbe890dc3
	.long	0xbd8f4709
	.long	0x3cde5bf9
	.long	0x3e961e06
	.long	0x3e78208b
	.long	0x3de86706
	.long	0x3dc2c69f
	.long	0xbb944f7c
	.long	0x3d8d80cb
	.long	0x3cf23c33
	.long	0xbd06f518
	.long	0xbdab5964
	.long	0xbd2016d6
	.long	0x3d14204b
	.long	0xbd763f66
	.long	0x3daffd50
	.long	0xbd8d7e41
	.long	0x3d1ac0af
	.long	0x3de7ab16
	.long	0xbb952cd1
	.long	0xbd549d00
	.long	0xbd871101
	.long	0xbd1dd7ae
	.long	0xbc920a64
	.long	0xbe5dcd9b
	.long	0xbe878a48
	.long	0xbe6fcf75
	.long	0xbd0d61e1
	.long	0x3dc878e7
	.long	0x3e3b4d6b
	.long	0x3e2418e9
	.long	0x3e7b3342
	.long	0x3d334280
	.long	0x3e4a17f6
	.long	0x3d96d932
	.long	0x3e659547
	.long	0xbd75dc93
	.long	0xbc8bdb40
	.long	0xbdc63fc1
	.long	0xbdc39c09
	.long	0x3bc51128
	.long	0xbe009d36
	.long	0x3dc639eb
	.long	0x3e2468ca
	.long	0xbdc28233
	.long	0xbe87a7fe
	.long	0xbe50e32f
	.long	0x3c6175bb
	.long	0x3d1c461e
	.long	0xbdbfe2f1
	.long	0xbe21072a
	.long	0xbd9230ca
	.long	0xbddb3fce
	.long	0xbd08e7dc
	.long	0x3d181f07
	.long	0xbdb8ddf8
	.long	0xbdac9048
	.long	0xbe00de71
	.long	0xbe1ac6ca
	.long	0x3be46e1a
	.long	0x3d5565af
	.long	0x3caad6da
	.long	0x3de63e10
	.long	0xbdd452e3
	.long	0x3d192c60
	.long	0x3d874ed3
	.long	0x3cdec08a
	.long	0x3c62311c
	.long	0xbde32c05
	.long	0xbe6cfdc2
	.long	0x3d3bc22d
	.long	0x3da888e3
	.long	0xbe66dafb
	.long	0xbdd200db
	.long	0xbddf2399
	.long	0xbdc86602
	.long	0xbcfb82aa
	.long	0x3e41a5a5
	.long	0x3d35de22
	.long	0xbc9a8bc1
	.long	0xbe0a884f
	.long	0xbe2f4033
	.long	0x3e39047c
	.long	0x3ddd5655
	.long	0xbc82ce27
	.long	0xbe0b6000
	.long	0xbe5f1d68
	.long	0xbdf2da75
	.long	0xbdb81f55
	.long	0xbe031448
	.long	0xbd75d11f
	.long	0xbdbea433
	.long	0xbe28d4bb
	.long	0xbd8d7fac
	.long	0x3d66fd54
	.long	0x3da8804f
	.long	0xbe01e7ba
	.long	0x3d9dfcea
	.long	0x3ea73aa7
	.long	0x3e23091b
	.long	0xbd25aa9d
	.long	0xbe00e2c0
	.long	0x3e8639bd
	.long	0x3ea3f4bf
	.long	0x3def7327
	.long	0x3e97be03
	.long	0x3ed2d48c
	.long	0xbd82d345
	.long	0x3dc53e69
	.long	0xbdd0c24e
	.long	0xbd94108f
	.long	0x3c91af20
	.long	0xbdf9fb0e
	.long	0x3d8cf5aa
	.long	0xbea4fbc7
	.long	0xbee5118a
	.long	0xbe59434f
	.long	0xbd3381a0
	.long	0xbe528137
	.long	0xbe4e3164
	.long	0xbe67c5c8
	.long	0xbea66a5a
	.long	0xbd3dd790
	.long	0xbe1cfa06
	.long	0xbe35e040
	.long	0xbdb64bf0
	.long	0xbea6ff03
	.long	0x3dd8eed6
	.long	0x3e961ae2
	.long	0x3dbcd3c7
	.long	0xbddc0244
	.long	0xbdef354f
	.long	0x3de51c70
	.long	0x3e247d54
	.long	0x3e43ab32
	.long	0xbc8ad303
	.long	0xbdba1292
	.long	0x3c0e06b6
	.long	0xbdb44685
	.long	0x3e626a2b
	.long	0x3ea2d7d4
	.long	0xbe0df3f0
	.long	0x3d77af54
	.long	0x3c249c97
	.long	0x3d666a92
	.long	0x3e6503f9
	.long	0x3d279fad
	.long	0xbb8449ba
	.long	0x3a49e31b
	.long	0x3e8f7a06
	.long	0x3cf47312
	.long	0xbd0b4c97
	.long	0xbe526fb1
	.long	0xbe5d31bb
	.long	0xbed2ce7c
	.long	0xbe4a3e77
	.long	0xbe5d964c
	.long	0xbe060b03
	.long	0xbe63f425
	.long	0xbe816e4e
	.long	0xbd895365
	.long	0xbe03efee
	.long	0x3ce5c83a
	.long	0x3d891046
	.long	0xbca9b60c
	.long	0xbdaf7e7a
	.long	0x3e2a214e
	.long	0x3e0b0016
	.long	0x3d551ec2
	.long	0xbe1abf0d
	.long	0xbe33cec5
	.long	0x3e51210b
	.long	0xbd63c738
	.long	0xbe1ee4b4
	.long	0xbe8151b5
	.long	0x3d36b690
	.long	0x3e2950d6
	.long	0x3e9f9683
	.long	0x3c3a8413
	.long	0x3d7641a5
	.long	0xbe1b74a4
	.long	0x3ce0024e
	.long	0xbd3d2efc
	.long	0xbdcb7653
	.long	0x3db9333d
	.long	0xbdfab515
	.long	0xbd66dd35
	.long	0xbdca464b
	.long	0xbce67957
	.long	0xbdc6cc97
	.long	0xbbeb9153
	.long	0xbe383608
	.long	0xbd831cc5
	.long	0xbcde874e
	.long	0xbe8ab554
	.long	0xbe387a3f
	.long	0xbebe1792
	.long	0x3c7bf54e
	.long	0xbe61038f
	.long	0xbc93bfba
	.long	0xbe1d6578
	.long	0xbe2a815e
	.long	0xbe03bf5b
	.long	0xbe8a8df2
	.long	0xbd280f04
	.long	0x3e22645f
	.long	0x3de563f2
	.long	0x3da4f46b
	.long	0xbe498234
	.long	0xbeb5ffb0
	.long	0x3e0bc1ae
	.long	0x3e5f5e51
	.long	0x3cf755b0
	.long	0xbc3a93b1
	.long	0xbe68f154
	.long	0xbc813b76
	.long	0x3e2d4248
	.long	0x3e1d61e7
	.long	0x3d80e58d
	.long	0xbe20341a
	.long	0x3c501792
	.long	0x3d0f9d65
	.long	0xbcd379b7
	.long	0xbcc5ba43
	.long	0xbc399bbc
	.long	0x3d25f4ac
	.long	0x3bffab5e
	.long	0x3dd9d42e
	.long	0xbe2ae2e5
	.long	0xbeafdd13
	.long	0xbda2c56d
	.long	0xbd882608
	.long	0xbbc16b2f
	.long	0xbe2a2310
	.long	0xbebee05e
	.long	0xbe5f84bf
	.long	0x3dfa074e
	.long	0x3d92ab05
	.long	0xbdeeffa0
	.long	0xbe856876
	.long	0xbde445f2
	.long	0x3eabe29c
	.long	0x3dc30538
	.long	0x3dbe38de
	.long	0xbab26f0f
	.long	0x3d3cc987
	.long	0x3dde72ce
	.long	0xbd8ccf26
	.long	0x3d9211d1
	.long	0x3d1fc368
	.long	0xbc837a94
	.long	0xbd20b641
	.long	0xbe1e9801
	.long	0x3dffa429
	.long	0x3e569ab7
	.long	0xbc0943ce
	.long	0x3c8539b2
	.long	0x3e07f463
	.long	0x3d891f22
	.long	0xbe121a9b
	.long	0x3d4510ca
	.long	0x3e22dd15
	.long	0x3e2b57df
	.long	0xbdf12059
	.long	0xbe107507
	.long	0xbd8d7487
	.long	0x3dfc82cc
	.long	0xbdcd7a8f
	.long	0xbd075008
	.long	0xbd921c77
	.long	0xbd514956
	.long	0xbe5c5f2d
	.long	0xbe217d4b
	.long	0x3defc863
	.long	0x3929b73b
	.long	0x3e16dba9
	.long	0xbc6c9eed
	.long	0x3d9dc71b
	.long	0xbc060217
	.long	0xbdc2d523
	.long	0xbcabec85
	.long	0x3dfcfbe2
	.long	0x3e055c2b
	.long	0x3d9e5680
	.long	0x3db63bdc
	.long	0x3d5caa5d
	.long	0x3d4a9d02
	.long	0xbd533b1c
	.long	0x3d24aba0
	.long	0x3d76980e
	.long	0x3da41e3c
	.long	0x3e4cbf60
	.long	0x3e3cfc71
	.long	0xbd81a30c
	.long	0xbc787153
	.long	0x3dcab7e0
	.long	0x3e4f1c79
	.long	0xbdc2eab2
	.long	0xbe240142
	.long	0xbe6c2cb1
	.long	0xbd8b7aa9
	.long	0xbd9bd6b3
	.long	0x3d9bd916
	.long	0x3d32f8de
	.long	0xbd7672e3
	.long	0xbe5d2988
	.long	0xbe1d1a88
	.long	0x3d4b35ab
	.long	0x3c61ce37
	.long	0xbe0dd034
	.long	0x3e006ccd
	.long	0x3dbb6048
	.long	0xbdad78bd
	.long	0xbd997338
	.long	0x3cbe8f78
	.long	0x3e92e414
	.long	0x3e947df7
	.long	0xbdc9d5da
	.long	0xbdcb2023
	.long	0x3e454b4d
	.long	0xbc7c7ec7
	.long	0x3d8ce7a5
	.long	0xbe363ff1
	.long	0x3c6551fb
	.long	0xbc18fc9e
	.long	0xbecedac0
	.long	0xbeba1c65
	.long	0x3cc90233
	.long	0xbdefa652
	.long	0xbe4990ac
	.long	0xbe789e6e
	.long	0xbe970197
	.long	0xbdc38541
	.long	0xbe98fd72
	.long	0xbe562efb
	.long	0xbd8430d2
	.long	0xbdbf8991
	.long	0xbe05c435
	.long	0xbd9e1893
	.long	0xbc7e1c1f
	.long	0xbe08621e
	.long	0x3dcdf554
	.long	0xbe3ca386
	.long	0x3def3a83
	.long	0x3e0e383b
	.long	0xbe06292f
	.long	0x3caa7964
	.long	0x3c438aee
	.long	0x3c88c3ac
	.long	0xbdfc7cd2
	.long	0xbe318758
	.long	0x3ddad425
	.long	0xbe533dcc
	.long	0xbb2741d9
	.long	0xbda44d47
	.long	0xbc82657f
	.long	0x3e5f4063
	.long	0xbe4aa8f7
	.long	0xbe2b6043
	.long	0xbd8323c6
	.long	0xbe023504
	.long	0x3e075c35
	.long	0xbe1936d2
	.long	0xbdd8e74f
	.long	0xbd8645c2
	.long	0x3dc4c5cb
	.long	0x3ee142fa
	.long	0xbdec728e
	.long	0xbe43f0d5
	.long	0xbe297bca
	.long	0xbe052c52
	.long	0xbd72c381
	.long	0xbe522328
	.long	0xbe99d227
	.long	0xbe0fe039
	.long	0x3c52df7e
	.long	0x3e259af6
	.long	0x3d6765a1
	.long	0xbe5d50e6
	.long	0xbe121f3c
	.long	0xbd464fcb
	.long	0x3d851503
	.long	0xbccb775b
	.long	0xbe56977d
	.long	0xbe07f8c1
	.long	0x3e553975
	.long	0x3e98a9ec
	.long	0xbecaab7c
	.long	0x3aa16c6f
	.long	0x3e247c33
	.long	0x3e18cbfc
	.long	0xbe301278
	.long	0xbe658e76
	.long	0xbdfab0c5
	.long	0x3cfbf68e
	.long	0xbdaefb22
	.long	0xbe75e88f
	.long	0xbe542d71
	.long	0xbde73d16
	.long	0xbd9f4d16
	.long	0xbefb8428
	.long	0xbe857a7c
	.long	0x3dc076e8
	.long	0xbd7748af
	.long	0xbe301328
	.long	0xbdc23c06
	.long	0x3e7152cd
	.long	0x3cf4f040
	.long	0xbe08f5e2
	.long	0xbdef2d7a
	.long	0x3d827320
	.long	0x3e468d41
	.long	0x3c9546e9
	.long	0xbde66a8e
	.long	0xbdb86ffd
	.long	0xbb74c086
	.long	0xbd2a0c6a
	.long	0x3ce2bc70
	.long	0xbd8950a9
	.long	0xbddd55f4
	.long	0x3d95160a
	.long	0x39c4ddb7
	.long	0x3e0cbeec
	.long	0xbda34255
	.long	0x3d4db594
	.long	0xbd98b4fb
	.long	0x3d14a0a2
	.long	0x3d89c598
	.long	0x3ca0bcf0
	.long	0xbbe6b2da
	.long	0xbb211d62
	.long	0x3e1618ee
	.long	0xbdb8969d
	.long	0xbc8cf1fe
	.long	0xbb42de5a
	.long	0x3d2695e6
	.long	0x3e0876bb
	.long	0xbe454004
	.long	0x3d1dc0e5
	.long	0x3d9b9a65
	.long	0xbd1f7fa7
	.long	0x3e508bb3
	.long	0xbd5f96bb
	.long	0xbe07fbe2
	.long	0xbe2edc64
	.long	0x3daa26f7
	.long	0x3d133e54
	.long	0x3e071e5f
	.long	0x3d67efe0
	.long	0xbe177c01
	.long	0xbd52a7d1
	.long	0x3ccc125b
	.long	0x3dc28136
	.long	0x3db3a41e
	.long	0xbe3cef81
	.long	0xbea7a894
	.long	0xbe30776e
	.long	0x3e7fb859
	.long	0x3de3faff
	.long	0xbdfa6296
	.long	0xbe888218
	.long	0xbd63cda4
	.long	0x3ead46c5
	.long	0xbc300052
	.long	0xbe0aca3f
	.long	0xbcbf2f39
	.long	0xbe226503
	.long	0x3e081a97
	.long	0xbe2a97be
	.long	0x3d16165e
	.long	0x3eba01b1
	.long	0xbbad4fd9
	.long	0xbdb6feee
	.long	0xbe2bdb8a
	.long	0x3ed4b806
	.long	0x3dac100a
	.long	0xbbb3e831
	.long	0xbe30e88f
	.long	0xbe35f6ac
	.long	0xbe8234ea
	.long	0xbe447249
	.long	0xbe5c12ad
	.long	0x3cf8cda5
	.long	0xbe9d2789
	.long	0xbe170591
	.long	0xbe2715f5
	.long	0xbcb9a5eb
	.long	0xbe22b9c8
	.long	0xbeb0a2af
	.long	0x3c87ca48
	.long	0x3e82cc4e
	.long	0x3ed6d168
	.long	0xbe350260
	.long	0xbe26dd0b
	.long	0x3e190e72
	.long	0x3d4ebc71
	.long	0x3d2dc23f
	.long	0xbea7163d
	.long	0xbca23ca7
	.long	0x3db0cdb7
	.long	0x3c38d468
	.long	0xbd31c4d6
	.long	0x3cd5a0cb
	.long	0x3dbce5f4
	.long	0xbc0ad02a
	.long	0xbcbb9438
	.long	0xbde006b0
	.long	0x3d438ecc
	.long	0x3db0cba3
	.long	0xbdb221a5
	.long	0xbdd3f4a9
	.long	0xbe72d2ed
	.long	0x3c792573
	.long	0x3e9bd680
	.long	0x3d07ede1
	.long	0xbe2bc422
	.long	0x3ddcf320
	.long	0x3e125371
	.long	0x3e222111
	.long	0xbdd01391
	.long	0x3e671ecd
	.long	0x3d9c569f
	.long	0x3d115052
	.long	0xbc136216
	.long	0x3e7ac688
	.long	0x3d328623
	.long	0x3e07e319
	.long	0xbdee6e54
	.long	0x3d405af6
	.long	0xbe00a4bd
	.long	0xbe05960c
	.long	0x3e8499e1
	.long	0x3e15c299
	.long	0x3c64deac
	.long	0xbe392ded
	.long	0xbdfb3a60
	.long	0x3e5de46a
	.long	0x3e059045
	.long	0xbe7de6a0
	.long	0xbd9ed3c6
	.long	0x3dd8af1c
	.long	0x3eb0d191
	.long	0x3e1bf56f
	.long	0xbe38b6b3
	.long	0x3d9c3ad5
	.long	0xbd25cb97
	.long	0x3ea194eb
	.long	0xbe714a6e
	.long	0xbdca465b
	.long	0xbcaa2d3b
	.long	0x3e03c025
	.long	0x3bff6eff
	.long	0xbd640f48
	.long	0xbd26ad1b
	.long	0x3c94474d
	.long	0x3d9bb3dd
	.long	0xbd17f0b6
	.long	0xbe466a8d
	.long	0xbe8c409f
	.long	0x3d703829
	.long	0x3ba2a071
	.long	0xbdc70dc0
	.long	0xbe4964d0
	.long	0xbd9da982
	.long	0x3d62e718
	.long	0xbd150e33
	.long	0xbe8341b1
	.long	0x3e122487
	.long	0x3d8bd259
	.long	0xbd0cc025
	.long	0xbe5cf84b
	.long	0xbe4d4615
	.long	0x3e327b63
	.long	0xbc218203
	.long	0xbe313fbc
	.long	0x3c81f6f2
	.long	0xbd0e8d97
	.long	0x3d9618d7
	.long	0x3e17cc92
	.long	0x3e670700
	.long	0x3dc02aea
	.long	0x3ccbb9c5
	.long	0x3cd6944a
	.long	0x3d406ed6
	.long	0x3c34c1ae
	.long	0xbde86c68
	.long	0x3e053a65
	.long	0x3cf2014c
	.long	0x3da75de6
	.long	0xbe347dd5
	.long	0xbe82db9b
	.long	0x3dcd127d
	.long	0xbde7264e
	.long	0xbe69355d
	.long	0xbe835913
	.long	0x3e4a040d
	.long	0x3e981e01
	.long	0xbe0a7b8f
	.long	0xbe2e5bc5
	.long	0xbe3d5b96
	.long	0x3d72a79f
	.long	0x3da974cc
	.long	0xbdfa6e82
	.long	0x3df00bb9
	.long	0x3e7a0eb0
	.long	0x3eadd7bd
	.long	0xbc24b87b
	.long	0xbe52abe6
	.long	0xbde10d03
	.long	0xbc830216
	.long	0xbdb0be46
	.long	0xbe601839
	.long	0xba9ce63a
	.long	0xbdc674a3
	.long	0xbe927608
	.long	0xbe8146d7
	.long	0xbe971275
	.long	0xbe4b8ec6
	.long	0xbe7d1ba3
	.long	0xbe84edb2
	.long	0xbdd903aa
	.long	0xbe1b71ed
	.long	0xbe07d8ee
	.long	0xbeb1fd6c
	.long	0xbeb4d866
	.long	0xbcdcb084
	.long	0xbd390652
	.long	0x3e09fb3e
	.long	0x3d862eb5
	.long	0x3df048ac
	.long	0x3e3c94fa
	.long	0x3e516e9f
	.long	0xbe1f54d4
	.long	0xbe0d6457
	.long	0xbd3da8c2
	.long	0x3d828b9d
	.long	0x3d4a3d97
	.long	0xbe392937
	.long	0xbd780ee9
	.long	0xbe6a9043
	.long	0xbde3885b
	.long	0xbe8a6d4b
	.long	0xbdaa8994
	.long	0xbe36d4d8
	.long	0xbe8cdba6
	.long	0xbe31eb83
	.long	0xbd6bb341
	.long	0xbe82e254
	.long	0xbeb05ccc
	.long	0xbeaa0803
	.long	0xbe079776
	.long	0xbe1a5943
	.long	0xbd3bd79e
	.long	0xbe3897e3
	.long	0xbca0269d
	.long	0x3dbe0df1
	.long	0x3c9e7280
	.long	0xbe57e6a2
	.long	0x3cbb4530
	.long	0xbd094d45
	.long	0xbe0f3d18
	.long	0xbd8e79f5
	.long	0xbe33837a
	.long	0x3e041653
	.long	0x3dc2eb05
	.long	0xbb2d4ed2
	.long	0xbe0e8e31
	.long	0xbdc7532c
	.long	0x3e18c4f5
	.long	0x3d9fc33d
	.long	0xbd82622d
	.long	0xbd00ca30
	.long	0xbe4d3d03
	.long	0x3d7efba2
	.long	0xbe1d7c4d
	.long	0x3e0f55e9
	.long	0x3d128eb4
	.long	0xbe911809
	.long	0x3da0d78c
	.long	0x3d066463
	.long	0x3c99dbaa
	.long	0xbdfc9541
	.long	0xbd98280f
	.long	0x3e2aa62d
	.long	0x3e35a3db
	.long	0x3e1a136d
	.long	0xbe375651
	.long	0x3d9630a9
	.long	0x3e1173ad
	.long	0xbc237b68
	.long	0xbe5ab625
	.long	0xbd48991e
	.long	0x3e41a908
	.long	0xba037a35
	.long	0xbe78ace3
	.long	0xbe1521bf
	.long	0xbd085c2f
	.long	0x3dd8aede
	.long	0xbea18536
	.long	0xbe1aa391
	.long	0x3b530110
	.long	0x3e17a5c8
	.long	0xbe6582bd
	.long	0xbd818d1e
	.long	0xbe59b99b
	.long	0x3d2e4b78
	.long	0x3dbae492
	.long	0xbe2c4215
	.long	0xbbe7ea50
	.long	0xbe99e7bb
	.long	0xbe725dd3
	.long	0xbe415b5e
	.long	0xbd7d136f
	.long	0x3d857d80
	.long	0xbe9f6e61
	.long	0xbe2b339f
	.long	0xbe657263
	.long	0xbdc0892b
	.long	0xbdc16b63
	.long	0x3e53fc47
	.long	0x3f085c3f
	.long	0x3dd2cdf1
	.long	0xbc0e22b5
	.long	0x3e63e0a7
	.long	0x3ec5597d
	.long	0x3d817f0c
	.long	0x3d3ff5a9
	.long	0xbe754f6e
	.long	0x3e18a33a
	.long	0x3d848688
	.long	0x3e3d847a
	.long	0x3db3a413
	.long	0xbc8d02cb
	.long	0xbdd9be28
	.long	0x3e154ec3
	.long	0x3e93f570
	.long	0x3ddff96c
	.long	0xbcdea224
	.long	0xbcac2abc
	.long	0xbd8871f9
	.long	0x3e6eea80
	.long	0x3e1afc39
	.long	0xbe40e402
	.long	0xbc8e8c54
	.long	0x3dd8ebb4
	.long	0xbe8a43f7
	.long	0xbda53467
	.long	0x3dd8346b
	.long	0xbd369c3a
	.long	0xbe8581b2
	.long	0xbe3ace1d
	.long	0x3ddfc1d7
	.long	0xbe258cbc
	.long	0x3d356490
	.long	0x3dcf73a5
	.long	0x3dc42120
	.long	0xbc4b2384
	.long	0xbde2fc31
	.long	0xbd7a4d7b
	.long	0x3c9b8eb8
	.long	0xbdc484f4
	.long	0xbebe743d
	.long	0x3df718be
	.long	0x3e215f5c
	.long	0x3e3e98ab
	.long	0xbe53d709
	.long	0xbe4c9b5f
	.long	0x3e4f7b8c
	.long	0x3ddb1d05
	.long	0xbe14e390
	.long	0xbe8a3b3a
	.long	0xbe3b9d90
	.long	0xbd9bb208
	.long	0xbdd388bd
	.long	0xbe99c5bc
	.long	0xbe79a258
	.long	0xbe1d8761
	.long	0xbe5cb1f9
	.long	0x3c1e9657
	.long	0x3d43b318
	.long	0xbc1cd1ac
	.long	0xbe17f625
	.long	0xbe329da0
	.long	0xbd51c6de
	.long	0x3ceaf5be
	.long	0xbc1b0fd0
	.long	0xbe33fbee
	.long	0xbdb8439a
	.long	0xbdcbb26d
	.long	0xbca498bd
	.long	0xbd832a02
	.long	0xbb5dc6a8
	.long	0xbe4310b1
	.long	0x3bf10348
	.long	0x3d5503c7
	.long	0x3e4074a1
	.long	0x3ec0a399
	.long	0x3e062e0c
	.long	0x3e6c23b9
	.long	0x3e8c9172
	.long	0x3e44106a
	.long	0xbd9cf1a9
	.long	0xbca8fefe
	.long	0x3dd2e982
	.long	0xbbdb77e1
	.long	0xbcde7b9a
	.long	0x3e930ccc
	.long	0xbd160f8a
	.long	0x3e66039d
	.long	0x3e3ac22b
	.long	0xbe23b8e8
	.long	0x3d86198b
	.long	0xbdd642ec
	.long	0xbe3e8416
	.long	0xbd958b4f
	.long	0xbd2f5aa9
	.long	0x3daba0a4
	.long	0xbea5ae05
	.long	0x3dc582b9
	.long	0xbe058ab1
	.long	0x3d2eef60
	.long	0xbe10cca7
	.long	0xbd60abcc
	.long	0xbd88f1c0
	.long	0xbd32a28b
	.long	0xbc3c240a
	.long	0xbde0bcb0
	.long	0xbca796e4
	.long	0xba9cffe1
	.long	0xbcca1160
	.long	0xbd990251
	.long	0x3ea0acc6
	.long	0xbcd2c518
	.long	0x3e311b77
	.long	0x3e2d2389
	.long	0x3d5f3289
	.long	0x3e6f4a7f
	.long	0x3d52848a
	.long	0xbda73d1a
	.long	0xbba18cd7
	.long	0xbdd5cb8c
	.long	0xbdba6f06
	.long	0xbe05843b
	.long	0x3cca3322
	.long	0x3dbe1311
	.long	0xbdb79db2
	.long	0xbe5e8b7d
	.long	0xbdf2b5c9
	.long	0x3dab3925
	.long	0xbe1404e3
	.long	0xbe996a0e
	.long	0xbd717007
	.long	0x3e541c01
	.long	0xbc010e39
	.long	0x3e1e526a
	.long	0x3c09118c
	.long	0x3d901d5c
	.long	0xbddb73ad
	.long	0xbd479c26
	.long	0xbc855f3c
	.long	0x3deb310f
	.long	0x3d40f554
	.long	0xbdb2b40b
	.long	0xbe7d15bb
	.long	0xbea105e7
	.long	0xbe59e973
	.long	0xbc88d359
	.long	0xbed5c404
	.long	0xbeaa0261
	.long	0xbef20d00
	.long	0xbee10092
	.long	0xbe949c68
	.long	0xbeb028f2
	.long	0xbebc3db8
	.long	0xbea565c9
	.long	0xbdd25d45
	.long	0x3df10b25
	.long	0xbe4b34f6
	.long	0xbdb96a97
	.long	0xbc692288
	.long	0xbda690ea
	.long	0xbd1163dd
	.long	0xbdc325ef
	.long	0xbd66714b
	.long	0x3db12a00
	.long	0xbe100825
	.long	0xbdd42149
	.long	0x3e1b5fd1
	.long	0x3e26183e
	.long	0xbdd33760
	.long	0xbdb033a0
	.long	0xbe4b05e9
	.long	0xbcca4dd3
	.long	0xbd86f305
	.long	0x3d4d35e5
	.long	0x3dd30c73
	.long	0x3e1f37d2
	.long	0xbe433bb8
	.long	0xbdd3d21f
	.long	0x3e16a1fc
	.long	0x3dcafa9c
	.long	0x3cd551f5
	.long	0xbe744022
	.long	0xbcb4727f
	.long	0xbe8e6ff7
	.long	0xbe1a578a
	.long	0x3e276f89
	.long	0x3db904d5
	.long	0x3d611826
	.long	0xbe599979
	.long	0x3d33cc1a
	.long	0x3e5f6c8d
	.long	0x3e0fdf5b
	.long	0x3b401560
	.long	0xbdd8592a
	.long	0x3e044bbc
	.long	0x3c3480f2
	.long	0x3e75eb12
	.long	0xbeaa2530
	.long	0xbdb86371
	.long	0xbe15b151
	.long	0xbe5f301e
	.long	0x3c968dab
	.long	0xbdcc8aa2
	.long	0xbebb9683
	.long	0xbed5a8ed
	.long	0xbdee5448
	.long	0xbcab4290
	.long	0x3df34e5e
	.long	0x3e33b4ed
	.long	0x3d2042a6
	.long	0xbe391cc1
	.long	0x3bc6bc0b
	.long	0xbd813861
	.long	0x3df28c8f
	.long	0x3ed372b6
	.long	0xbddf701f
	.long	0x3dd2db3e
	.long	0xbe21ef0c
	.long	0x3e1169bc
	.long	0x3e989a13
	.long	0x3cb8b239
	.long	0xbe1c0fa5
	.long	0xbba0f2e6
	.long	0x3e054399
	.long	0x3ee6dfd5
	.long	0x3d93f7e8
	.long	0xbe057fa9
	.long	0xbe09813d
	.long	0x3b636b5a
	.long	0x3e5443a9
	.long	0x3c15f66d
	.long	0x3dd013e0
	.long	0xbc919408
	.long	0xbdd285a6
	.long	0xbd8753b5
	.long	0xbdb1548b
	.long	0x3df9f985
	.long	0x3e4592f9
	.long	0xbcd41713
	.long	0xbc96d0f9
	.long	0xbe34b749
	.long	0xbdac1ba1
	.long	0xbcf154a8
	.long	0xbe473c23
	.long	0xbdfd2211
	.long	0xbeb9c84d
	.long	0xbdeacfef
	.long	0xbdd15ba3
	.long	0xbdbe0919
	.long	0x3e192b8f
	.long	0xbe1c3a00
	.long	0xbe685623
	.long	0xbe1625f7
	.long	0xbe191df1
	.long	0x3dcc7a5d
	.long	0x3c957cf0
	.long	0x3e709907
	.long	0x3ddd36d4
	.long	0xbd0c768c
	.long	0xbd5574ae
	.long	0xbd8d9257
	.long	0x3e0afd00
	.long	0x3e28ab87
	.long	0xbe467682
	.long	0x3df1a348
	.long	0xbdb3dd7e
	.long	0x3d3f7ec9
	.long	0xbe1e71ee
	.long	0xbdf79bdd
	.long	0x3ca1ea7e
	.long	0xbe289cd4
	.long	0x3d032069
	.long	0xbe928ab4
	.long	0xbe8309a4
	.long	0x3ca6cf62
	.long	0xbdeb0f56
	.long	0xbd53d27b
	.long	0xbe554935
	.long	0xbd4cd3ce
	.long	0xbd89378f
	.long	0xbe863b30
	.long	0x3d2493bb
	.long	0xbe370f65
	.long	0xbdc403ba
	.long	0xbdc6e981
	.long	0xbd84ca37
	.long	0x3d50012e
	.long	0x3da59cb0
	.long	0xbe10a1ad
	.long	0xbe173e7a
	.long	0x3dd93d0a
	.long	0xbdedeb47
	.long	0xbdc2c3af
	.long	0xbe8a868e
	.long	0xbcefd0a8
	.long	0x3e30267d
	.long	0xbdb6f359
	.long	0xbdeef3e7
	.long	0xbe1671de
	.long	0xbc7d176a
	.long	0x3ebedaba
	.long	0xbdd94ba3
	.long	0xbd95be1c
	.long	0xbc5dcdb2
	.long	0x3cc34705
	.long	0x3e0e8f5e
	.long	0x3db66143
	.long	0xbb577b3b
	.long	0xbe9e99e3
	.long	0xbe4eb749
	.long	0xbe0156e2
	.long	0x3dea6673
	.long	0x3cf28c51
	.long	0xbd5a35fc
	.long	0xbe8e0c66
	.long	0xbd486da5
	.long	0xbdb425f9
	.long	0xbdd79f1c
	.long	0xbd95dfdb
	.long	0xbe56a581
	.long	0xbe19a2c0
	.long	0x3c3fcd49
	.long	0xbe0732a1
	.long	0xbe0fbba9
	.long	0xbe0ff2df
	.long	0x3ded39fe
	.long	0xbcf8b111
	.long	0xbe38b1f4
	.long	0xbdd11f7f
	.long	0x3d83a3ff
	.long	0x3e3e32b6
	.long	0xbe504932
	.long	0xbe63fd13
	.long	0xbeb4d98c
	.long	0xbe352561
	.long	0x3e7a11d0
	.long	0xbdc6e844
	.long	0xbc5bed5a
	.long	0xbd55f2fa
	.long	0xbd8184f1
	.long	0x3e95413d
	.long	0xbd1bf6e4
	.long	0x3e231a68
	.long	0x3c2a403a
	.long	0x3d901adb
	.long	0x3ea42f22
	.long	0x3dd10070
	.long	0x3e94da16
	.long	0x3e158e0a
	.long	0x3e231aa8
	.long	0x3ebbf2c3
	.long	0x3e409818
	.long	0x3d902e19
	.long	0x3dc7b8e1
	.long	0xbdac704c
	.long	0x3c91cf8e
	.long	0x3dc297d8
	.long	0xbd640c09
	.long	0x3dba78bd
	.long	0x3d08bdd4
	.long	0x3d336b9e
	.long	0xbd434d1b
	.long	0x3e5e0b7a
	.long	0xbc7b7fed
	.long	0xbe13453e
	.long	0x3c99334f
	.long	0x3df88947
	.long	0x3d3250d4
	.long	0xbb846b17
	.long	0xbb80718c
	.long	0xbe3ff9b6
	.long	0x3d9c8ad5
	.long	0xbe5f4f24
	.long	0xbdff1636
	.long	0x3e33719b
	.long	0xbe381add
	.long	0xbd058113
	.long	0xbea5b695
	.long	0xbe5949c1
	.long	0xbe2a3238
	.long	0x3b860707
	.long	0x3d789142
	.long	0x3da02885
	.long	0xbd1707de
	.long	0xbc67a054
	.long	0xbddd1682
	.long	0x3ddc646b
	.long	0x3e19f719
	.long	0xbe22be91
	.long	0xbdc433a6
	.long	0xbd8f7edb
	.long	0xbd480569
	.long	0xbe021dd8
	.long	0x3cca2c72
	.long	0x3cdec6a8
	.long	0xbcf77e5d
	.long	0xbe57a7ad
	.long	0x3cd8b0db
	.long	0x3e09cb14
	.long	0x3decc445
	.long	0xbd4715d8
	.long	0x3d18cf48
	.long	0x3e8925dd
	.long	0x3dd04135
	.long	0x3e1e1cec
	.long	0x3eaa8aa9
	.long	0x3c9c266a
	.long	0xbd8d94df
	.long	0xbdb6d25d
	.long	0xbe0d9c57
	.long	0xbd4bac54
	.long	0x3d8abddf
	.long	0xbe1a534d
	.long	0xbba75afa
	.long	0x3c81e4e4
	.long	0xbd34518a
	.long	0x3dd46448
	.long	0xbe6ce992
	.long	0xbdc7f37f
	.long	0xbd981db7
	.long	0x3d72b79e
	.long	0x3e38399d
	.long	0xbca75226
	.long	0x3d557e9c
	.long	0x3d9d0465
	.long	0xbd9b0680
	.long	0x3dfb42d4
	.long	0x3d3c5bcf
	.long	0x3de5c282
	.long	0x3e28a59c
	.long	0xbb4e76e2
	.long	0xbdaf53f6
	.long	0xbe2f8e47
	.long	0xbe39e661
	.long	0xbe4cc1c2
	.long	0xbce1a5d2
	.long	0xbdd61aa0
	.long	0xbe212f7e
	.long	0xbda079ea
	.long	0xbd5c0080
	.long	0x3c90389f
	.long	0xbd386a2a
	.long	0xbcd0b9f6
	.long	0xbd7b6f2c
	.long	0x3cae6d8c
	.long	0x3d867eac
	.long	0xbe03215c
	.long	0x3d9e2f8b
	.long	0x3e296e3d
	.long	0x3d14f319
	.long	0xbe6baf5a
	.long	0x3e1152f1
	.long	0x3e69ac9a
	.long	0x3ea32dc2
	.long	0xbe3f902d
	.long	0xbec841ca
	.long	0xb998c789
	.long	0xbd4d10a3
	.long	0xbdeb2949
	.long	0xbdf8f789
	.long	0x3dfde549
	.long	0x3b9de501
	.long	0xbd220663
	.long	0xbd2ee396
	.long	0x3d903aaa
	.long	0xbc3376c1
	.long	0x3d6f3cb5
	.long	0xbcc5f202
	.long	0x3db31e95
	.long	0x3cee2f41
	.long	0x3dbfc803
	.long	0xbda6fb09
	.long	0x3cc64694
	.long	0xbd5bb1ad
	.long	0xbd910860
	.long	0xbd732fff
	.long	0xbe778c42
	.long	0xbe0d2f68
	.long	0xbe81e5ad
	.long	0xbdd493e2
	.long	0x3db96284
	.long	0xbbe02c08
	.long	0xbe3ffdd5
	.long	0xbe62325c
	.long	0xbdbbfe41
	.long	0xbb6ee0fc
	.long	0xbe4d0d91
	.long	0xbe042de3
	.long	0xbdc1dbfd
	.long	0x3b27b5bd
	.long	0x3c1d19ea
	.long	0x3d4904cb
	.long	0x3e851e03
	.long	0x3d007e40
	.long	0xbd8aa0cf
	.long	0xbd6b2acd
	.long	0x3e427962
	.long	0x3c515c37
	.long	0xbe09359f
	.long	0xbe4df3d8
	.long	0xbde19589
	.long	0xbe2356e9
	.long	0xbd4d140e
	.long	0xbe7d81a1
	.long	0xbe2733da
	.long	0x3d723edd
	.long	0xbd4438c7
	.long	0xbc9f3ac5
	.long	0xbe2d7306
	.long	0x3bb97a41
	.long	0xbd9aedb1
	.long	0xbe2c2563
	.long	0xbdb48064
	.long	0xbe2a07fd
	.long	0xbde1e47b
	.long	0xbd1b296e
	.long	0x3e2c4220
	.long	0x3e58ff27
	.long	0x3d6d1f9a
	.long	0xbd0e6558
	.long	0xbdd2a743
	.long	0x3ddad722
	.long	0x3d3ffb65
	.long	0xbd440749
	.long	0xbe43b897
	.long	0xbdf05d9e
	.long	0xbe43f30f
	.long	0xbe0533f9
	.long	0xbe2543eb
	.long	0xbe8c6887
	.long	0xbde97617
	.long	0xbb16b7b5
	.long	0xbd09c8c5
	.long	0xbc2c83c8
	.long	0xbe255d02
	.long	0xbd09e678
	.long	0x3c499e5d
	.long	0x3d028056
	.long	0x3de79909
	.long	0xbd985d98
	.long	0xbd10258d
	.long	0xbce2c6ac
	.long	0x3e0c04ce
	.long	0xbb82a1eb
	.long	0xbdf5afb3
	.long	0x3b4ed1ea
	.long	0x3d4b171d
	.long	0xbd0c3c5d
	.long	0xbd837930
	.long	0xbdcbfc5a
	.long	0xbda23da6
	.long	0xbdcc7d92
	.long	0xbcba9936
	.long	0x3d068dfa
	.long	0xbd9d51d4
	.long	0x3d819a59
	.long	0xbdbdad02
	.long	0xbe6f3fba
	.long	0xbe3411c0
	.long	0xbdf87252
	.long	0x3df20fd7
	.long	0x38ce5f34
	.long	0xbe2d330f
	.long	0xbe12a0ce
	.long	0xbdb64249
	.long	0xbdbf57d3
	.long	0xbdb96224
	.long	0xbd99c026
	.long	0xbda497d8
	.long	0xbcc4ea8b
	.long	0xbce77ef0
	.long	0x3c9f2e77
	.long	0x3e436cd0
	.long	0x3e512e85
	.long	0x3a9664cd
	.long	0xbdef7788
	.long	0x3e7131ad
	.long	0x3e625633
	.long	0x3db362dc
	.long	0xbe81d07f
	.long	0xbda34191
	.long	0xbe755eac
	.long	0x3e0efd72
	.long	0x3e6a5faa
	.long	0xbe3573ac
	.long	0xbd835704
	.long	0xbe3065f4
	.long	0xbd46c201
	.long	0xbe00b905
	.long	0xbd9d1e10
	.long	0xbd89fb0f
	.long	0xbdf933ad
	.long	0xbd77a7d2
	.long	0x3d996f67
	.long	0xbdab5a06
	.long	0xbe240a30
	.long	0x3dc63c87
	.long	0x3cd64152
	.long	0xbced28fc
	.long	0x3de68bfd
	.long	0x3cad348b
	.long	0x3e71724b
	.long	0x3e0d87c5
	.long	0xbe304081
	.long	0x3e834d5c
	.long	0x3e19e9f0
	.long	0xbe1db737
	.long	0xbba815c3
	.long	0xbdd6d866
	.long	0x3d01bf95
	.long	0xbe0354e9
	.long	0x3c839d56
	.long	0xbdccf960
	.long	0x3d2746dd
	.long	0x3daee7fb
	.long	0xbe8167a6
	.long	0x3c9a6ef5
	.long	0xbb701c9e
	.long	0xbca999ea
	.long	0xbd1a1853
	.long	0x3db7bc9f
	.long	0xbdf61da8
	.long	0xbd8bf874
	.long	0xbd9b8ece
	.long	0x3bda05f0
	.long	0x3dff5ac0
	.long	0xbd0151cf
	.long	0xbe8e9555
	.long	0x3db4f1d4
	.long	0x3dbf00cd
	.long	0x3d539424
	.long	0x3e51c5b9
	.long	0x3e92e298
	.long	0x3d28915e
	.long	0xbe2da77b
	.long	0x3cb5c459
	.long	0xbd0dc2ef
	.long	0x3d9f5ce0
	.long	0x3cf62f95
	.long	0xbdf0f348
	.long	0xbe051c79
	.long	0x3d0ff5b1
	.long	0x3d7e981c
	.long	0x3e00f284
	.long	0xbdd45b67
	.long	0xbe7394d9
	.long	0x3c06ff21
	.long	0x3d74930f
	.long	0x3e493d78
	.long	0x3c7452d8
	.long	0xbd52c2a2
	.long	0x3c07f6f1
	.long	0x3cc4a97f
	.long	0xbe1d2083
	.long	0xbe68d2ec
	.long	0xbe32d9e1
	.long	0xbdb24bee
	.long	0xbda7d596
	.long	0x3ccdcd81
	.long	0xbe4bcc23
	.long	0xbe9f102d
	.long	0xbe865ed9
	.long	0xbe55387f
	.long	0xbe019736
	.long	0xbcc0fa7f
	.long	0xbe1f9294
	.long	0xbe16bbc0
	.long	0xbdd1b8da
	.long	0x3dd07eb0
	.long	0x3d232f6d
	.long	0x3e28ffd2
	.long	0xbd629033
	.long	0xbd215d22
	.long	0x3de37ba6
	.long	0x3d6f65d8
	.long	0x3eca2ce1
	.long	0x3d15116d
	.long	0x3cd08220
	.long	0x3e065312
	.long	0x3ed1c661
	.long	0x3ef0e45b
	.long	0xbbf5f867
	.long	0x3cdf15f8
	.long	0x3cc659c1
	.long	0xbcdb21a5
	.long	0xbe027fb3
	.long	0x3bddc39b
	.long	0x3dc3a347
	.long	0x3d78f691
	.long	0xbdb2d4c2
	.long	0x3ce513b1
	.long	0xbddf1961
	.long	0x3da55f9b
	.long	0x3c5bfd01
	.long	0x3d1aa44d
	.long	0x3e7c01d0
	.long	0xb7bfa77f
	.long	0xbdace936
	.long	0xbd59f7ee
	.long	0xbd7e6d40
	.long	0x3dba38b4
	.long	0x3c18ea88
	.long	0x3d2d90c6
	.long	0xbe24698c
	.long	0xbc2c6fc1
	.long	0xbd563736
	.long	0x3e31aef2
	.long	0x3d4d79ad
	.long	0xbdce137e
	.long	0xbc4c1d4f
	.long	0x3d49340b
	.long	0xbe1ee527
	.long	0x3d6364ae
	.long	0x3dc97438
	.long	0xbe072014
	.long	0x3cd62381
	.long	0x3ca83280
	.long	0x3d96fc79
	.long	0x3dfc327b
	.long	0xbd26eee2
	.long	0x3dfc2b55
	.long	0x3c90a5d1
	.long	0xbdc04823
	.long	0x3d300fed
	.long	0x3d1a68e5
	.long	0xbdda4652
	.long	0xbc70b7d8
	.long	0x3ceaa709
	.long	0xba9e96fb
	.long	0xbe0b9232
	.long	0xbebd9d20
	.long	0x3e0f56c2
	.long	0xbd926e0e
	.long	0x3e443ab6
	.long	0x3e585c52
	.long	0xbbf39d32
	.long	0x3db32970
	.long	0x3d0fda5a
	.long	0x3c8c144a
	.long	0x3c689859
	.long	0x3d284f39
	.long	0xbd72a86f
	.long	0xbc5d9a07
	.long	0xbdcc01b5
	.long	0xbe09daf4
	.long	0xbd5b3bbd
	.long	0x3d57ccaa
	.long	0xbe0c65ad
	.long	0xbdf99382
	.long	0xbe0221f4
	.long	0xbe01f778
	.long	0x3ddee046
	.long	0x3db9b574
	.long	0xbd7017e8
	.long	0xbd999d24
	.long	0xbdd54b78
	.long	0x3e5905a5
	.long	0x3e58ce04
	.long	0x3d8caae2
	.long	0xbe821923
	.long	0xbe24ecc8
	.long	0x3c8d3be4
	.long	0x3d2ffd88
	.long	0x3dddb7ea
	.long	0xbe26cf0f
	.long	0xbe7f6ef7
	.long	0x3aa4790e
	.long	0xbdbbeb92
	.long	0xbd91606d
	.long	0xbdcf81a2
	.long	0xbe59bbf9
	.long	0x3d6b4acb
	.long	0x3c63b831
	.long	0xba57f2c1
	.long	0xbe06d6aa
	.long	0xbe5da09b
	.long	0x3cda5299
	.long	0xbd236da0
	.long	0xbe567048
	.long	0xbdfb8aa3
	.long	0x3cd006ff
	.long	0x3d1b4ddc
	.long	0x3df0b15e
	.long	0xbdf5b33f
	.long	0xbe2e2553
	.long	0xbdc77cd2
	.long	0x3e50fdc7
	.long	0x3e41ec4f
	.long	0xbd2a432c
	.long	0x3d3fce8f
	.long	0xbd6065c4
	.long	0xbd9a0d9e
	.long	0xbda309a0
	.long	0xbdde6cfd
	.long	0xbe059d90
	.long	0xbc9c41ba
	.long	0xbe557b5a
	.long	0xbe010ae4
	.long	0xbd5441cc
	.long	0xbe093d47
	.long	0xbe83103b
	.long	0xbd41fcc6
	.long	0x3d1e88fa
	.long	0xbe25a18d
	.long	0xbe41ed8f
	.long	0xbe736614
	.long	0x3dc00d47
	.long	0x3d018453
	.long	0x3e30856f
	.long	0x3e1aeba9
	.long	0x3d88b9ce
	.long	0x3c8e2f3b
	.long	0xbe7e7025
	.long	0x3d8d8f0b
	.long	0x3da9d9ce
	.long	0x3e15c316
	.long	0x3e1800bd
	.long	0xbe2fb965
	.long	0xbdc15e82
	.long	0x3b4dcf6a
	.long	0x3d40dc9e
	.long	0x3d187429
	.long	0xbdb67b6c
	.long	0x3d0616b3
	.long	0xbe317c4b
	.long	0xbdfff896
	.long	0x3df5e72f
	.long	0xbc136b3d
	.long	0x3bbdb2a2
	.long	0xbe53a854
	.long	0x3d90a8ae
	.long	0xbb965afe
	.long	0xbdac7c5f
	.long	0x3d4423be
	.long	0x3c59701f
	.long	0x3d28f039
	.long	0xbd9f1169
	.long	0x3d35c485
	.long	0x3d9de25a
	.long	0xbb8186d4
	.long	0x3c3764d9
	.long	0x3e6c6238
	.long	0x3d105f3b
	.long	0xbe120b44
	.long	0xbbc99c46
	.long	0x3de27782
	.long	0x3e23a203
	.long	0xbdb3375f
	.long	0xbe4b02e4
	.long	0xbd4b416f
	.long	0xbac55e72
	.long	0xbd1d79e6
	.long	0xbd20e834
	.long	0x3c9064a9
	.long	0x3b554a97
	.long	0xbe3f104f
	.long	0x3e84cda7
	.long	0x3bf66256
	.long	0xbe736093
	.long	0xbe53bc28
	.long	0x3d49a2ae
	.long	0x3de95497
	.long	0xbe190824
	.long	0xbe9301ae
	.long	0xbe3d46c0
	.long	0xbcd78ef0
	.long	0xbe66520a
	.long	0xbe0e4254
	.long	0xbe532a72
	.long	0xbe0e60b4
	.long	0xbda3648b
	.long	0xbe424318
	.long	0xbe09dfb5
	.long	0xbe150bd6
	.long	0xbd6769a1
	.long	0xbe16a2de
	.long	0x3d4460f7
	.long	0x3d6e7ecf
	.long	0xbcacabac
	.long	0x3c83a029
	.long	0x3e3e7a7e
	.long	0x3d5c39d2
	.long	0x3c8a4027
	.long	0xbddc6263
	.long	0x3b1bc506
	.long	0xbe2a4799
	.long	0xbce30625
	.long	0xbd936480
	.long	0xbe15c3bd
	.long	0xbe4bb238
	.long	0x3c2191f2
	.long	0xbcff4966
	.long	0xbdacd452
	.long	0xbd694247
	.long	0x3cf92355
	.long	0xbe16e110
	.long	0xbc7a5c5d
	.long	0x3d31f959
	.long	0xbdcafba1
	.long	0x3d89a900
	.long	0xbe0c732b
	.long	0xbda047b5
	.long	0x3d797523
	.long	0x3cf2f004
	.long	0x3da5955a
	.long	0x3e60d349
	.long	0x3bff50aa
	.long	0xbe34f51f
	.long	0xbc1b0609
	.long	0xbc5b369b
	.long	0xbd6e0f0a
	.long	0xbe5f7a9e
	.long	0xbd359baa
	.long	0xbd0b1b8e
	.long	0xbe3571ad
	.long	0x3b884089
	.long	0xbd81ee71
	.long	0x3dc32fdb
	.long	0xbd740c22
	.long	0xbd2f8614
	.long	0xbd03ea93
	.long	0xbc8f8b54
	.long	0x3c5f1e7c
	.long	0x3d426f2e
	.long	0xbd375be3
	.long	0xbe09960d
	.long	0xbd2f0f9d
	.long	0xbd07d243
	.long	0xbd2871b8
	.long	0x3cf05761
	.long	0x3e7691e5
	.long	0xbe4e6583
	.long	0xbb84716e
	.long	0x3e1dd5f0
	.long	0xbe369e63
	.long	0xbb4013aa
	.long	0xbc752b11
	.long	0x3e46668f
	.long	0x3d0e9639
	.long	0xbd86217f
	.long	0x3d562536
	.long	0x3e80e9d3
	.long	0x3e1f94ef
	.long	0xbde42006
	.long	0xbd8d513a
	.long	0xbe03bfd1
	.long	0xbd8aabaa
	.long	0xbe62f389
	.long	0xbd9859ee
	.long	0x3cfa11bf
	.long	0xbe4ff237
	.long	0xbe63ef4a
	.long	0xbdec9cb7
	.long	0xbddfb9f1
	.long	0x3e373671
	.long	0x3e461691
	.long	0x3d96e901
	.long	0x3e667233
	.long	0xbd845a9f
	.long	0x3cd067e7
	.long	0x3da614b0
	.long	0x3e846dee
	.long	0x3e0a87cc
	.long	0xbc0282c3
	.long	0x3c489957
	.long	0x3dcf7806
	.long	0x3e70390c
	.long	0xbdc73f3b
	.long	0xbe86ddda
	.long	0xbd949540
	.long	0x3d7c7358
	.long	0xbdc35c20
	.long	0xbe40fa2c
	.long	0xbded9e58
	.long	0xbe1b66c1
	.long	0xbd9dc4c2
	.long	0xbdb76259
	.long	0x3c949395
	.long	0xbcc8dc24
	.long	0xbdceb3fe
	.long	0xbe56ad44
	.long	0x3dbf1180
	.long	0xbd1f426e
	.long	0xbe54675a
	.long	0xbe482a40
	.long	0xbe348761
	.long	0xbdf02a33
	.long	0x3e25ed06
	.long	0xbe37abe0
	.long	0xbe6325f8
	.long	0x3d6a6657
	.long	0xbd92beee
	.long	0x3bd4dbe3
	.long	0x3e510315
	.long	0x3f109191
	.long	0x3e4c526f
	.long	0xbe078887
	.long	0x3e94944f
	.long	0x3e89ee29
	.long	0x3ea0a957
	.long	0x3deeb44f
	.long	0x3dd4fbc0
	.long	0x3da1bd08
	.long	0x3e853a79
	.long	0x3e255e78
	.long	0xbd48b3b5
	.long	0xbdbbdfa1
	.long	0xbd672723
	.long	0xbdfe773c
	.long	0xbe866d29
	.long	0xbe0fc5c7
	.long	0x3d082c0c
	.long	0xbd7a79d3
	.long	0x3c219003
	.long	0x3e722bbf
	.long	0xbe4b6288
	.long	0x3d8957b1
	.long	0x3d0377c9
	.long	0x3ea5fcdc
	.long	0x3e1a0da7
	.long	0xbc9068f4
	.long	0x3e4c7507
	.long	0xbe33e316
	.long	0x3e206914
	.long	0xbdede117
	.long	0xbe2f0fbe
	.long	0x3b5f29ac
	.long	0xbde86409
	.long	0x3d012ad3
	.long	0xbe086cea
	.long	0xbe56ddf5
	.long	0x3c059e9a
	.long	0xbc960ab3
	.long	0xbd149e55
	.long	0xbe16c4ab
	.long	0xbe65b321
	.long	0x3df90ac3
	.long	0x3dc06cd8
	.long	0x3d1bbf1e
	.long	0xbd93a276
	.long	0xbe9b13f4
	.long	0x3dc021c1
	.long	0x3e046843
	.long	0xbdf41ca3
	.long	0xbe50f12e
	.long	0xbdcd358d
	.long	0x3daa5032
	.long	0xbe958a6f
	.long	0xbea63487
	.long	0xbe0ce3e0
	.long	0x3d9e8c69
	.long	0x3d6872c6
	.long	0x3d457adc
	.long	0xbe2af670
	.long	0xbe2c59bf
	.long	0xbe83a8b9
	.long	0x3cb44276
	.long	0x3db8b358
	.long	0x3dc69796
	.long	0x3db90781
	.long	0xbe045b0a
	.long	0x3b8ab115
	.long	0xbd1416dd
	.long	0x3c8a7f2a
	.long	0xbe2f1d3b
	.long	0xbe031815
	.long	0x3b21570b
	.long	0xbe030369
	.long	0xbe87e624
	.long	0xbe690afe
	.long	0xbd4e72fc
	.long	0x3d47586b
	.long	0xbd48a1a6
	.long	0x3e08d273
	.long	0x3da76a19
	.long	0xbd937887
	.long	0x3dbdd0d3
	.long	0x3d58dd2d
	.long	0x3dd1ce58
	.long	0x3c31ede9
	.long	0xbdeff38c
	.long	0x3c51e28e
	.long	0x3e09ef35
	.long	0x3dec7424
	.long	0xbe3ac43d
	.long	0xbd9941f5
	.long	0x3e13fcea
	.long	0x3d446ab7
	.long	0xbdb3e3e9
	.long	0xbe251ad6
	.long	0xbd110ff0
	.long	0x3be0ba99
	.long	0xbe51b022
	.long	0xbdad2f5e
	.long	0xbcc1617e
	.long	0xbdbf8020
	.long	0xbd8dd168
	.long	0xbe0f2805
	.long	0xbdd61072
	.long	0x3dee4a28
	.long	0x3dad23d4
	.long	0x3c7356c2
	.long	0xbe081474
	.long	0xbd2884b7
	.long	0x3e09b709
	.long	0x3cd90dc8
	.long	0x3c2b25a9
	.long	0x3df99485
	.long	0x3df94872
	.long	0xbd8aae03
	.long	0x3d4653fc
	.long	0x3dea63e4
	.long	0x3d96cedb
	.long	0x3d49ad16
	.long	0xbdaa3a56
	.long	0xbdea7f39
	.long	0xbc50f456
	.long	0xbdf9f7ba
	.long	0xbdb455be
	.long	0x3e29e343
	.long	0xbe0128f7
	.long	0x3e1d2047
	.long	0xbdd091cc
	.long	0xbe91e041
	.long	0xbe49f0e4
	.long	0x3e149e3a
	.long	0xbdef925e
	.long	0xbe225540
	.long	0xbe717a3f
	.long	0xbe12463b
	.long	0x3da8d5cd
	.long	0x3bcb2d42
	.long	0x3cfde912
	.long	0xbe563092
	.long	0xbe47bae3
	.long	0xbd15639a
	.long	0xbe218f75
	.long	0xbe4a412c
	.long	0xbea4aad6
	.long	0xbe730343
	.long	0x3cfa0d13
	.long	0xbe9fdff8
	.long	0xbeaa2a4a
	.long	0xbeb66115
	.long	0x3d95e085
	.long	0x3e50ceaf
	.long	0xbe2c27a3
	.long	0xbe7d8174
	.long	0xbe5dc45a
	.long	0x3e94f063
	.long	0x3ecc1fd5
	.long	0xbdf8fa1e
	.long	0x3d33148a
	.long	0x3ea7b1f4
	.long	0x3eb9331a
	.long	0xbe811c18
	.long	0x3ddadba9
	.long	0xbce02a1a
	.long	0xbe4971ac
	.long	0xbe4e1ce2
	.long	0xbe82480a
	.long	0x3dc72a44
	.long	0x3e6cafda
	.long	0x3d93972c
	.long	0x3dfdba04
	.long	0x3df3a15c
	.long	0x3e214333
	.long	0x3cf70b77
	.long	0xbdb4a4b5
	.long	0x3e2edada
	.long	0x3e227ecc
	.long	0xbe018693
	.long	0xbe54dcbc
	.long	0xbd4d6428
	.long	0x3dd449fe
	.long	0xbe10310e
	.long	0x3e154fd7
	.long	0xbda1bce3
	.long	0xbdbe26ad
	.long	0xbcfdd360
	.long	0xbdb1e836
	.long	0x3e599aa4
	.long	0x3d760fdb
	.long	0x3da2ea77
	.long	0xbd464c14
	.long	0xbcec9fd5
	.long	0xbd1e11f4
	.long	0xbcca8595
	.long	0xbdec029d
	.long	0x3d2342de
	.long	0xbd02d9be
	.long	0xbe73f7ae
	.long	0xbd0da4af
	.long	0xbdc716f4
	.long	0x3d88e026
	.long	0xbc8b2448
	.long	0x3dce2d87
	.long	0xbdd6f289
	.long	0xbe0e253b
	.long	0xbe11848b
	.long	0xbe8dbbf9
	.long	0x3ea0a3f8
	.long	0xbd2b5946
	.long	0xbe5aae8e
	.long	0xbd28ba6a
	.long	0xbe04e311
	.long	0xbd768e11
	.long	0x3d45148d
	.long	0x3db6440b
	.long	0x3ddacdb7
	.long	0x3e6e85b9
	.long	0xbe272fe7
	.long	0xbe6c16c3
	.long	0xbe1e1a9f
	.long	0xbd9c9d61
	.long	0xbb994361
	.long	0x3d6201b0
	.long	0xbd993c89
	.long	0xbccd5cde
	.long	0x3e2d4791
	.long	0x3dab1909
	.long	0x3e368d30
	.long	0x3d5bf63a
	.long	0x3cf80385
	.long	0x3e130e66
	.long	0x3dcb4b1c
	.long	0x3e875649
	.long	0x3d09a11b
	.long	0x3c0a4735
	.long	0x3e8e5664
	.long	0x3e5e1404
	.long	0xbd9a0f5e
	.long	0x3e09f245
	.long	0x3da2c94c
	.long	0x3e576cf8
	.long	0x3e614cd2
	.long	0xbdc8f83d
	.long	0xbd813afe
	.long	0xbc22bbf4
	.long	0xbe390c81
	.long	0xbe4895b8
	.long	0xbe3c24bc
	.long	0xbde07057
	.long	0x3c9bed42
	.long	0x3e11f60d
	.long	0x3e2a43ef
	.long	0xbd138e67
	.long	0xbdd1944a
	.long	0xbd734b49
	.long	0xbd35bd3e
	.long	0x3bbd02d2
	.long	0xbd00768a
	.long	0xbe0e19bc
	.long	0xbe5646be
	.long	0xbe8b46ef
	.long	0xbe3a1d14
	.long	0xbdc506cd
	.long	0x3d2a332b
	.long	0x398e775d
	.long	0xbdb9094d
	.long	0xbdb6c6b9
	.long	0x3dada6c2
	.long	0xbcb602e4
	.long	0xbdc63e98
	.long	0x3bf03afd
	.long	0x3d2fc547
	.long	0xbe109234
	.long	0xbe378ad3
	.long	0xbe41e9d7
	.long	0xbda0ff4d
	.long	0x3df87393
	.long	0x3cb18515
	.long	0xbd9ce7c3
	.long	0xbe8f47de
	.long	0xbdc55836
	.long	0x3e79c78b
	.long	0x3e3b30e2
	.long	0xbc99ce45
	.long	0xbe8022f5
	.long	0xbc6ecd47
	.long	0x3df0bf54
	.long	0xbe96dbe1
	.long	0xbc930280
	.long	0xbd418b3c
	.long	0xbe28197a
	.long	0xbd887b44
	.long	0x3d958647
	.long	0xbd3e867f
	.long	0xbddf34f8
	.long	0x3b821c92
	.long	0x3daea7ef
	.long	0x3d2c7ee3
	.long	0x3e40210a
	.long	0xbe4cbf2f
	.long	0xbe181161
	.long	0xbcc95138
	.long	0xbe57c894
	.long	0x3db644d6
	.long	0x3d5f6a5c
	.long	0xbde3b521
	.long	0x3e34eb53
	.long	0xbd9a88f8
	.long	0x3d45e149
	.long	0xbe5bdeb4
	.long	0xbdca2bea
	.long	0x3e3dd1cb
	.long	0xbe3d0301
	.long	0xbda935c9
	.long	0xbcdaabd1
	.long	0xbddccfcb
	.long	0xbdcefbe4
	.long	0x3d593e97
	.long	0xbe21c726
	.long	0xbd82ed24
	.long	0x3dc9f4fc
	.long	0x3cf4f349
	.long	0x3ac722d9
	.long	0x3e85754a
	.long	0x3d662613
	.long	0xbe6e1bcb
	.long	0x3b9e71ba
	.long	0xbe32ff25
	.long	0x3e49f69f
	.long	0x3d8b6a32
	.long	0x3a85699a
	.long	0x3d7f540b
	.long	0xbdfd4af3
	.long	0xbe13a130
	.long	0xbe18b024
	.long	0xbe3c6ad7
	.long	0x3e15b618
	.long	0xbe3f65d5
	.long	0xbe328f5e
	.long	0x3b876107
	.long	0xbe0eb7e8
	.long	0xbd5e338c
	.long	0x3cbb5d00
	.long	0xbd5fc6c2
	.long	0x3c34898f
	.long	0xbdff3b79
	.long	0xbd25bf28
	.long	0x3e5a168b
	.long	0x3dc9492d
	.long	0x3dabf060
	.long	0xbd800fcd
	.long	0xbdd7a743
	.long	0x3a3afdd9
	.long	0x3e94e514
	.long	0x3dbf6959
	.long	0xbd3cffcf
	.long	0x3cac4611
	.long	0x3da3c05e
	.long	0x3ebf8d3b
	.long	0x3d562b09
	.long	0xbe2c6432
	.long	0x3da78124
	.long	0x3de8efa9
	.long	0x3e1ca4cd
	.long	0x3e70db71
	.long	0x3e3f9393
	.long	0x3e0c906c
	.long	0xbded3d43
	.long	0xbe01ed32
	.long	0xbe5def54
	.long	0xbe274a17
	.long	0xbe6a807a
	.long	0xbe43cf13
	.long	0xbdc83d76
	.long	0xbd2a1592
	.long	0xbd52e693
	.long	0xbcee187b
	.long	0xbdca0667
	.long	0xbdb9887a
	.long	0xbe4abf0e
	.long	0xbd8796c7
	.long	0xbdec0ab3
	.long	0x3da7c198
	.long	0x3d1ad8e1
	.long	0xbe79d526
	.long	0xbe1ff591
	.long	0xbd11d4b4
	.long	0xbe87ef0f
	.long	0xbcd51948
	.long	0x3d082f2b
	.long	0xbe15cf83
	.long	0x3d1fb011
	.long	0xbea2dd6e
	.long	0x3e10b945
	.long	0x3e5ecd79
	.long	0xbcbf7d8e
	.long	0xbe4a79df
	.long	0xbe05071a
	.long	0x3e30766a
	.long	0x3e31e967
	.long	0xbcdaa8df
	.long	0xbd74b517
	.long	0x3e1a71a8
	.long	0x3e355a6d
	.long	0xbd674bc8
	.long	0xbe472813
	.long	0xbde68b6c
	.long	0x3dd0e71d
	.long	0x3d835720
	.long	0x3cb771be
	.long	0xbd475618
	.long	0xbd899ff4
	.long	0xbe243fdf
	.long	0x3e990238
	.long	0x3dc0733e
	.long	0xbd5487d7
	.long	0xbe520a64
	.long	0xbd8a0b0d
	.long	0x3ec084a8
	.long	0x3d918639
	.long	0xbdb8507c
	.long	0xbdb5b8f0
	.long	0x3de13816
	.long	0x3e9f46db
	.long	0xbd231b78
	.long	0xbe30f419
	.long	0xbdb4a991
	.long	0x3da84726
	.long	0x3e354ac2
	.long	0xbc9242fd
	.long	0xbe85fd9a
	.long	0xbd265e8b
	.long	0x3e0e3139
	.long	0xbd80e65a
	.long	0xbe6f8b11
	.long	0xbe2b3212
	.long	0xbd861d20
	.long	0xbda9ad4c
	.long	0x3c531d99
	.long	0x3cfb681e
	.long	0xbe3827a7
	.long	0x3c7f0048
	.long	0x3d4d36c5
	.long	0xbdddc479
	.long	0xbe3ab7c5
	.long	0xbd4a3cab
	.long	0x3df4a40d
	.long	0xbe7ae2bb
	.long	0xbe358b3a
	.long	0xbe83d1a7
	.long	0xbe020639
	.long	0x3de0119d
	.long	0xbe4964d7
	.long	0xbe3b04db
	.long	0xbdfae548
	.long	0xbdfa1228
	.long	0xbe43e4db
	.long	0xbe9ca0ab
	.long	0xbdbb6ea8
	.long	0xbd6c5576
	.long	0x3c057323
	.long	0xbdf42c0f
	.long	0xbea8cf91
	.long	0x3d4176f8
	.long	0x3d8162f5
	.long	0x3d517ebf
	.long	0xbe09ba04
	.long	0xbdaa40e7
	.long	0x3df73cd2
	.long	0x3dd0d342
	.long	0xbe74aec2
	.long	0xbdc8e228
	.long	0xbd0140ed
	.long	0x3e9b8a3e
	.long	0x3da38a48
	.long	0xbe758327
	.long	0xbe364ea7
	.long	0xbc9126b2
	.long	0x3e8178f4
	.long	0x3a7d3891
	.long	0xbe8f5223
	.long	0xbdab82f2
	.long	0x3e9696a6
	.long	0x3c728eab
	.long	0xbdd9ab8c
	.long	0xbe732ad5
	.long	0xbc8539af
	.long	0xbeb216ab
	.long	0xbed5c69a
	.long	0xbd0c73f7
	.long	0xbcf61f1f
	.long	0xbe256ac9
	.long	0xbe900b37
	.long	0xbe1703af
	.long	0x3d7a94f6
	.long	0x3bec88ba
	.long	0xbe59bdda
	.long	0xbd581032
	.long	0x3d072ee8
	.long	0x3d50f0a1
	.long	0xbdb65918
	.long	0xbd9cba59
	.long	0xbe1287d1
	.long	0x3d374b10
	.long	0x3d225a2e
	.long	0xbe03b539
	.long	0xbc9e7b30
	.long	0xbdb456d6
	.long	0x3d053385
	.long	0xbde5a32e
	.long	0x3c618393
	.long	0xbe3d368f
	.long	0xbe6a1c59
	.long	0xbea11ef6
	.long	0xbea26b30
	.long	0xbdc92958
	.long	0x3c44dd95
	.long	0xbdb053e1
	.long	0xbe7fd9b6
	.long	0xbcd3372b
	.long	0xbe21bfe4
	.long	0x3c27124b
	.long	0x3c809af4
	.long	0x3d323f95
	.long	0xbc6baa0d
	.long	0x3cd3c04f
	.long	0x3e0f958e
	.long	0xbe320d1c
	.long	0xbc0f58a2
	.long	0xbde46512
	.long	0x3de8be7b
	.long	0xbd8bf8d8
	.long	0xbe813b83
	.long	0xbd868f78
	.long	0xbde51954
	.long	0xbdcfde10
	.long	0xbe129ae1
	.long	0xbdf6b1ac
	.long	0xbe699af3
	.long	0xbe773a10
	.long	0xbdb39da1
	.long	0xbe09ab63
	.long	0x3bc2b634
	.long	0xbe9d8e6b
	.long	0xbc5a541c
	.long	0x3d05d5b5
	.long	0xbcc1a2f4
	.long	0x3dc7257a
	.long	0xbe51def7
	.long	0x3d2f3dc0
	.long	0x3e122d35
	.long	0x3dbc0e85
	.long	0xbe3f4362
	.long	0x3e164eaa
	.long	0x3e2a87b4
	.long	0x3e06bd61
	.long	0x3c21b91c
	.long	0xbe25aa75
	.long	0x3eba62b9
	.long	0x3e5083e9
	.long	0x3dc3143a
	.long	0xbb643c7e
	.long	0xbe16310e
	.long	0x3bf6ba01
	.long	0x3d851526
	.long	0x3d746681
	.long	0x3d8c65d2
	.long	0xbe9df16b
	.long	0xbe7e9f77
	.long	0x3c6edaf2
	.long	0x3e07f9d5
	.long	0x3e49efe6
	.long	0xbedd5fce
	.long	0x3d8e1a43
	.long	0xbd856eec
	.long	0xbe1071b2
	.long	0xbdc0f8c7
	.long	0xbdd1122e
	.long	0xbbc249a7
	.long	0xbae58d02
	.long	0xbe55ecc2
	.long	0xbdc2c23f
	.long	0xbe55927b
	.long	0x3d52f268
	.long	0xbe101c9e
	.long	0x3c5fbd92
	.long	0xbde94118
	.size	.L__constant_16x8x5x5xf32, 12800

	.type	.L__constant_16x4x4x10xf32,@object
	.p2align	6, 0x0
.L__constant_16x4x4x10xf32:
	.long	0x3dbbaa05
	.long	0x3df8b363
	.long	0x3daecc52
	.long	0x3dfb194e
	.long	0xbdca3f86
	.long	0xbde5a186
	.long	0xbd125c9a
	.long	0xbdc52512
	.long	0xbe176fc6
	.long	0x3d0ec83f
	.long	0xbd466fdd
	.long	0x3e0c2d95
	.long	0x3e158f5d
	.long	0xbdf5499d
	.long	0x3e550894
	.long	0xbe6895a2
	.long	0x3ed8bf7c
	.long	0x3d705782
	.long	0xbe2b54dd
	.long	0xbebddd39
	.long	0xbe44da5d
	.long	0x3dc12aea
	.long	0x3dbfe17d
	.long	0xbc2af9ea
	.long	0x3e6c2064
	.long	0xbee7a87c
	.long	0x3b23e4b1
	.long	0xbc4bbd56
	.long	0x3d0229d8
	.long	0xbe9bcc4f
	.long	0x3c93ab33
	.long	0x3cc6e9f8
	.long	0x3e931fef
	.long	0x3e2f1820
	.long	0xbd985c31
	.long	0x3c90bb7b
	.long	0xbdd7c5be
	.long	0xbda78cb8
	.long	0x3d7543fb
	.long	0x39f83eb3
	.long	0xbe1d3136
	.long	0x3df70c74
	.long	0x3ddc7cca
	.long	0x3dc173e5
	.long	0xbd8f0dc1
	.long	0xbcd265f4
	.long	0x3ddf8028
	.long	0xbdaa58ef
	.long	0x3da91f35
	.long	0xbe33ecc3
	.long	0xbe2c3431
	.long	0x3dd75392
	.long	0xbe327f45
	.long	0xbda88396
	.long	0x3eedf39d
	.long	0x3cf7e50c
	.long	0xbe2329ba
	.long	0x3e4c5f69
	.long	0xbdceef78
	.long	0xbefd3472
	.long	0x3ea8bfe6
	.long	0x3e35c522
	.long	0x3ed2a0ce
	.long	0x3dfc706c
	.long	0xbe8d2ffd
	.long	0xbeeebe1a
	.long	0xbebe17f1
	.long	0xbcabfcdf
	.long	0xbe0312ad
	.long	0x3d46e722
	.long	0x3ec369bc
	.long	0xbe6d8bcd
	.long	0x3ed0d61c
	.long	0x3cb99ded
	.long	0x3d9f7622
	.long	0xbf2b9132
	.long	0xbd7a93d1
	.long	0x3d3f8efe
	.long	0xbd06ea13
	.long	0x3e11a96a
	.long	0xbe03a3ee
	.long	0x3cd4c828
	.long	0xbc12c417
	.long	0x3e37b28d
	.long	0xbd090db6
	.long	0xbcde7b4b
	.long	0x3e056c67
	.long	0xbd3a5636
	.long	0x3d5bdf04
	.long	0x3cb2fe28
	.long	0xbbc37c6a
	.long	0xbd4b6fec
	.long	0xbe22ba4b
	.long	0x3ddee93a
	.long	0xbe2e25d5
	.long	0x3eb2fb32
	.long	0xbe327c49
	.long	0x3e29734c
	.long	0xbe0fbd2f
	.long	0xbe31e74c
	.long	0xbd15ebb1
	.long	0x3d2443a0
	.long	0x3e449079
	.long	0x3d2e25ed
	.long	0xbe27c1d6
	.long	0x3ee2427b
	.long	0x3d9da086
	.long	0xbee012b3
	.long	0x3d4aa982
	.long	0xbe716dab
	.long	0xbdc93aad
	.long	0xbecf99f6
	.long	0xbebefcc0
	.long	0x3e7989b9
	.long	0xbf05874d
	.long	0x3f111fa0
	.long	0x3f0e230c
	.long	0xbdd5df2b
	.long	0x3e385930
	.long	0xbde8fb91
	.long	0xbd22680e
	.long	0xbde5ab53
	.long	0xbb1b5739
	.long	0xbb4e0ba5
	.long	0xbdd27817
	.long	0x3c9bfb32
	.long	0xbd93bef5
	.long	0xbd80ecbd
	.long	0x3db1ffc6
	.long	0x3d73d1e4
	.long	0x3e38b714
	.long	0x3ddcefa7
	.long	0xbe48b869
	.long	0x3c035781
	.long	0x3d81a3a3
	.long	0x3c728543
	.long	0xbd933c80
	.long	0x3ddae231
	.long	0xbe7aed8b
	.long	0x3c79e400
	.long	0xbeab9d02
	.long	0xbd064c98
	.long	0xbe22e3aa
	.long	0x3f15900e
	.long	0xbd968bd0
	.long	0x3f040779
	.long	0x3e2502a6
	.long	0xbe4d38a8
	.long	0x3db05ade
	.long	0xbe8891e3
	.long	0xbef70cf5
	.long	0x3d25b931
	.long	0xbcd40a39
	.long	0x3e995956
	.long	0xbe5ede45
	.long	0x3da63940
	.long	0x3e887a99
	.long	0xbe949250
	.long	0x3e887f95
	.long	0xbe4fecc0
	.long	0xbdfe8fc7
	.long	0x3bc37991
	.long	0x3e65bf5d
	.long	0x3e0bd16a
	.long	0xbce599a7
	.long	0xbe8fd496
	.long	0x3cbae754
	.long	0x3d8557d6
	.long	0xbe158bea
	.long	0x3dd12645
	.long	0xbe0c8b26
	.long	0x3d65cbd3
	.long	0xbda620e6
	.long	0x3be90c54
	.long	0xbe53a5ff
	.long	0x3de7d5aa
	.long	0xbe387560
	.long	0x3eb83a3a
	.long	0xbe6b5463
	.long	0x3bfec4fb
	.long	0xbe59c1e0
	.long	0xbdd1efd2
	.long	0x3dba90c0
	.long	0x3e8aeea1
	.long	0xbe148657
	.long	0x3db2a760
	.long	0xbd8fcb96
	.long	0x3e0b183c
	.long	0x3df0cac1
	.long	0xbbc749dc
	.long	0xbd1e00c4
	.long	0x3db014d2
	.long	0x3d4232cc
	.long	0xbd1613f3
	.long	0x3da73efc
	.long	0x3dac7a14
	.long	0x3e09fa5a
	.long	0xbd811604
	.long	0x3bb5b830
	.long	0x3d9e4b52
	.long	0xbcc4c9fe
	.long	0xbd9742d8
	.long	0x3e787c7f
	.long	0x3e4d170a
	.long	0xbe0295c2
	.long	0xbec06c22
	.long	0xbe83c15e
	.long	0x3e2e5493
	.long	0xbd94edf4
	.long	0xb9dfb826
	.long	0xbd0f9f0e
	.long	0x3e812bf5
	.long	0x3e97b5cd
	.long	0x3e352939
	.long	0xbed07170
	.long	0xbe378c70
	.long	0xbef947fe
	.long	0x3f3f040a
	.long	0xbd321f1e
	.long	0xbd26d5e8
	.long	0x3d67dbbe
	.long	0xbe912286
	.long	0x3d8a521d
	.long	0xbe8caf9a
	.long	0xbe97342f
	.long	0x3ea38f71
	.long	0xbdcc552b
	.long	0x3eb7c1ba
	.long	0xbe0c65ae
	.long	0xbd78b1a2
	.long	0x3db6a884
	.long	0xbdfdbf0a
	.long	0xbef67b23
	.long	0xbdf7c780
	.long	0xbd9a956d
	.long	0x3f01ceaf
	.long	0x3e259620
	.long	0x3d090e0f
	.long	0xbc543230
	.long	0x3de14512
	.long	0xbe3987dc
	.long	0xbe5616ef
	.long	0x3e2ecc4a
	.long	0x3e55758e
	.long	0x3dd4df6e
	.long	0x3e28e6a4
	.long	0xbd54c055
	.long	0x3e987bfc
	.long	0xbdf562f2
	.long	0xbe358b25
	.long	0xbe120672
	.long	0xbb3e44ef
	.long	0xbe20b323
	.long	0xbd7a98fa
	.long	0x3e3c16c6
	.long	0x3e8c250d
	.long	0xbdb8872e
	.long	0xbdf271f7
	.long	0x3cdf4507
	.long	0x3eab2ca5
	.long	0xbe13c4c6
	.long	0xbe93e2a5
	.long	0xbe82910a
	.long	0xbdc738ea
	.long	0x3f13e8c6
	.long	0xbe9523f8
	.long	0x3e3e2f5c
	.long	0xbe6ec864
	.long	0x3e3a06bc
	.long	0xbc29f146
	.long	0xbd1c45f2
	.long	0xbdb69742
	.long	0x3f258685
	.long	0x3e2b6c81
	.long	0x3e01d867
	.long	0xbdc88e10
	.long	0x3e5b4e28
	.long	0x3e8770c6
	.long	0x3d5ca4e5
	.long	0xbf426f88
	.long	0xbe4fc676
	.long	0x3eb259e2
	.long	0xbe0cc4b7
	.long	0x3e92df57
	.long	0x3e5bef46
	.long	0x3dc9ca78
	.long	0xbe7630b6
	.long	0x3e235ee9
	.long	0xbe6836a9
	.long	0x3c8c952d
	.long	0x3c3ac424
	.long	0x3e3364f8
	.long	0x3d4c7e03
	.long	0x3d9e954d
	.long	0xbd3b9064
	.long	0x3cc85e2d
	.long	0xbd52b6c6
	.long	0xbefa3d10
	.long	0xbd86c4b5
	.long	0x3dce4e6e
	.long	0x3e896447
	.long	0xbd231dc8
	.long	0x3dc89002
	.long	0xbd222adf
	.long	0xbd9bd1df
	.long	0xb8daa9e7
	.long	0x3d05e262
	.long	0xbe9fb6b2
	.long	0x3db66d90
	.long	0xbe94578d
	.long	0xbda1c82f
	.long	0x3e7e4c19
	.long	0x3f27edc9
	.long	0xbe944660
	.long	0x3eb15380
	.long	0xbedaef08
	.long	0xbd801349
	.long	0x3d29922e
	.long	0xbdc2331b
	.long	0xbe1215f4
	.long	0x3d15565a
	.long	0xbe06ee5d
	.long	0x3c391c7b
	.long	0xbd2bd02a
	.long	0xbccfb759
	.long	0xbe358a02
	.long	0x3d4785bf
	.long	0x3be94bcf
	.long	0xbd0ffa08
	.long	0xbdf718cd
	.long	0x3d1ceb99
	.long	0xbe496c36
	.long	0x3e1bde64
	.long	0x3d8848ff
	.long	0x3cda4377
	.long	0xbe3a9b92
	.long	0xbe4d8fc0
	.long	0x3e7d93f2
	.long	0x3e122673
	.long	0xbc4eeaba
	.long	0x3e010b0a
	.long	0xbc94b4a6
	.long	0xbc3db2d9
	.long	0x3de8f29d
	.long	0xbeaab120
	.long	0xbe063da4
	.long	0xbe141fa0
	.long	0x3e6001cd
	.long	0x3e942cff
	.long	0x3e800b09
	.long	0x3e3224aa
	.long	0x3dd0d60e
	.long	0xbe832576
	.long	0xbde573ac
	.long	0xbd7233a8
	.long	0x3d103589
	.long	0xbd2736d7
	.long	0xbdb50bcc
	.long	0xbc8f5ad5
	.long	0x3d0f7eac
	.long	0xbdd3d48d
	.long	0x3d8e2e23
	.long	0x3e906396
	.long	0x3d884c49
	.long	0xbe2a7f84
	.long	0x3d87887e
	.long	0xbe287ccb
	.long	0x3ed12823
	.long	0x3d8032d1
	.long	0xbd824889
	.long	0xbe915a58
	.long	0x3e3b8742
	.long	0x3efd2b07
	.long	0x3e97ef68
	.long	0xbe8b19d7
	.long	0xbbfaf13d
	.long	0xbefe4a4f
	.long	0x3eafbce3
	.long	0x3dc2bbea
	.long	0xbe8ddc37
	.long	0x3e45ba4d
	.long	0xbe78f551
	.long	0x3e643f7d
	.long	0x3e38f529
	.long	0xbeae4e6e
	.long	0x3cc9809f
	.long	0xbdf60f2c
	.long	0x3eaac7cf
	.long	0xbe73b560
	.long	0x3cfb6cab
	.long	0x3caad132
	.long	0xbd6c188b
	.long	0xbc4b57f5
	.long	0xbe81ea3e
	.long	0xbeac994b
	.long	0x3efea0e4
	.long	0x3ecef0a1
	.long	0x3cb88843
	.long	0xbe6517bd
	.long	0x3b025a1d
	.long	0xbea3535f
	.long	0x3cf4b21b
	.long	0x3e339449
	.long	0x3e5b91f2
	.long	0x3e88c228
	.long	0x3de04bfd
	.long	0xbe90f042
	.long	0x3ea7c52a
	.long	0x3e8d1263
	.long	0xbec112f1
	.long	0xbf2c98ac
	.long	0xbed9350c
	.long	0xbe33e159
	.long	0x3e636888
	.long	0x3f069ef2
	.long	0x3eff3e29
	.long	0xbf3bf56f
	.long	0xbd946767
	.long	0xbd0221a6
	.long	0x3f1d3655
	.long	0x3e8cf4af
	.long	0xbe39eda7
	.long	0x3c30d21e
	.long	0x3e8e4053
	.long	0x3d474840
	.long	0x3e448cf3
	.long	0xbe11aacc
	.long	0x3dc0d37b
	.long	0xbe11a003
	.long	0xbd439ed4
	.long	0x3e0d80cb
	.long	0x3de90286
	.long	0xbb11a428
	.long	0x3c46f36a
	.long	0xbe031742
	.long	0x3e57e7c5
	.long	0x3dc9e25a
	.long	0xbbf0f1df
	.long	0xbdbb23d0
	.long	0xbe3e8ee7
	.long	0x3dbfd9c4
	.long	0x3e61253f
	.long	0x3c61cdde
	.long	0x3c97dfd9
	.long	0x3dcd3fa4
	.long	0x3d6b8a78
	.long	0x3db2ff8d
	.long	0xbe84b528
	.long	0x3d519f0e
	.long	0x3d01842d
	.long	0x3d9adb82
	.long	0xbe5cb001
	.long	0xbe3d7065
	.long	0x3e89d17b
	.long	0xbe6dddcf
	.long	0x3e75df98
	.long	0x3e5adf86
	.long	0xbf088215
	.long	0xbda14b08
	.long	0x3e58dbce
	.long	0x3e2ea575
	.long	0x3f0948a8
	.long	0xbe4c4d65
	.long	0xbbb9560b
	.long	0xbe47e95f
	.long	0xbdfe0021
	.long	0x3d8a668a
	.long	0xbdf9a2af
	.long	0x3d5f6e22
	.long	0xbe9a2181
	.long	0xba0fe813
	.long	0x3e2f9ea0
	.long	0x3f452ef3
	.long	0xbe8ab39e
	.long	0x3d714e15
	.long	0xbe12f309
	.long	0xbcc14e9c
	.long	0xbcb6461f
	.long	0xbe6fab42
	.long	0xbe8d30ed
	.long	0x3dce532d
	.long	0xbe082bd0
	.long	0xbe0af110
	.long	0x3e1a1fc5
	.long	0x3e0b2956
	.long	0xbdab4f9d
	.long	0x3e12ca15
	.long	0x3ddf3435
	.long	0xbde069c9
	.long	0x3b97f9ad
	.long	0x3da830f1
	.long	0xbd6962d7
	.long	0xbd058713
	.long	0x3e2afb32
	.long	0xbdeecf17
	.long	0x3e057824
	.long	0x3ded62a4
	.long	0x3dac1728
	.long	0xbe0f3373
	.long	0xbb8e2f48
	.long	0xbd9ab369
	.long	0x3d9a33c8
	.long	0x3e15b612
	.long	0xbd95d583
	.long	0xbe28e41e
	.long	0xbd455199
	.long	0x3de9f85e
	.long	0xbe29961b
	.long	0xbdcb68ce
	.long	0xbe314bf6
	.long	0x3dc8d514
	.long	0x3d566e50
	.long	0x3d12eb4f
	.long	0x39873767
	.long	0xbde2069a
	.long	0x3da2b407
	.long	0xbc698355
	.long	0x3d374ba4
	.long	0xbdc4ab0c
	.long	0xbd85700e
	.long	0x3c98de40
	.long	0x3dd8a6be
	.long	0x3de5e0ff
	.long	0x3e3161b7
	.long	0xbd864735
	.long	0x3d03ec55
	.long	0x3d877d4f
	.long	0xbdd2efc2
	.long	0xbd3adf35
	.long	0xbdb9eabb
	.long	0x3daf1674
	.long	0xbd8cf138
	.long	0x3ea8d4d6
	.long	0xbe0d56ee
	.long	0xbd80f713
	.long	0x3e3027cd
	.long	0xbd6bc44e
	.long	0xbe0eb64d
	.long	0xbe8e890e
	.long	0xbb6e9ade
	.long	0xbe2c8aea
	.long	0x3ecd40aa
	.long	0x3dd79933
	.long	0xbcf63923
	.long	0xbea0cbb3
	.long	0xbbd6edbf
	.long	0xbc9acd60
	.long	0xbcbd37bb
	.long	0xbe54f009
	.long	0x3dee62c2
	.long	0xbd3c45c6
	.long	0x3df8bc50
	.long	0x3df1b03b
	.long	0x3eb6e551
	.long	0xbe41099c
	.long	0xbe07e505
	.long	0x3e5d3424
	.long	0xbd91c903
	.long	0xbd76e6cd
	.long	0xbc6a6ec6
	.long	0x3dc8e697
	.long	0xbddd5590
	.long	0x3cfdb176
	.long	0xbdc7e75f
	.long	0x3cccc497
	.long	0xbe76cdae
	.long	0x3ce00da1
	.long	0x3b0a0f9b
	.long	0x3df44f30
	.long	0x3dc686d1
	.long	0x3ea35500
	.long	0xbe2a306f
	.long	0x3e27d954
	.long	0xbe7efc25
	.long	0x3d04a85f
	.long	0x3dc2a222
	.long	0x3d827ece
	.long	0x3ec95197
	.long	0xbecc6ac9
	.long	0xbe423856
	.long	0xbe79514e
	.long	0x3ebecbcd
	.long	0xbeb4ead8
	.long	0x3bae65c0
	.long	0xbd3818db
	.long	0x3d6f8a13
	.long	0xbe863fb4
	.long	0xbd723bda
	.long	0xbeb64408
	.long	0x3e53eed0
	.long	0x3e8a5b16
	.long	0x3e5ca899
	.long	0xbeb70006
	.long	0xbc1213f0
	.long	0xbecc4d8f
	.long	0x3c60b631
	.long	0x3cd8e444
	.long	0x3e119810
	.long	0xbe4f616d
	.long	0x3e8ea998
	.long	0xb9db2102
	.long	0xbe451e0e
	.long	0x3e22bf5a
	.long	0xbcbd4cba
	.long	0x3ce69527
	.long	0x3deea7dc
	.long	0xbc929af3
	.long	0xbd676837
	.long	0xbda3e750
	.long	0xbd971b51
	.long	0xbdd427d0
	.long	0x3d2ec7ba
	.long	0x3eb0eb04
	.long	0xbeb6025b
	.long	0x3d84311b
	.long	0xbdbe11b5
	.long	0x3dee1003
	.long	0x3e5b9324
	.long	0x3e935e7f
	.long	0xbeb853ca
	.long	0xbea0eb26
	.long	0x3db436a2
	.long	0xbecead7a
	.long	0x3d49dfff
	.long	0x3e85582c
	.long	0xbd3accb5
	.long	0xbdae4d80
	.long	0x3ec512f5
	.long	0x3e80ea15
	.long	0x3cf64df8
	.long	0xbe441340
	.long	0x3c2b612c
	.long	0xbe2836fb
	.long	0x3dcf8bb6
	.long	0x3d982c05
	.long	0x3d9294e7
	.long	0xbe094153
	.long	0xbe4458ba
	.long	0x3b9644a9
	.long	0x3c53f2ec
	.long	0xbc874706
	.long	0x3b329fcb
	.long	0xbc157e2b
	.long	0x3debbf1a
	.long	0x3d8c4c85
	.long	0xbdd3d3ac
	.long	0x3dedf68d
	.long	0x3c789d47
	.long	0x3ce160af
	.long	0xbcf16562
	.long	0x3db3dbb6
	.long	0x3e2e5773
	.long	0xbdda2fd9
	.long	0xbded85e8
	.long	0xbe0d9d2b
	.long	0xbc4f1b3a
	.long	0x3dd56e52
	.long	0xbd93e17c
	.long	0x3d2edfbc
	.long	0x3da9b3eb
	.long	0x3c8b0f46
	.long	0x3e83f92c
	.long	0xbde2392f
	.long	0xbe121aff
	.long	0x3db4859e
	.long	0x3e6c07dc
	.long	0xbd251cc0
	.long	0xbd99becc
	.long	0xbea1ad0e
	.long	0xbcb6c639
	.long	0xbe13cca7
	.long	0xbd94879d
	.long	0xbe222a5f
	.long	0xbe215a8b
	.long	0xbd87ea05
	.long	0x3eb54c0f
	.long	0x3be112f8
	.long	0x3d1ad051
	.long	0xbe8f8772
	.long	0x3c990587
	.long	0x3ca94cb4
	.long	0x3d8cc87b
	.long	0x3e4e8010
	.long	0x3dc52b30
	.long	0xbdf1ab5b
	.long	0x3df67847
	.long	0xbc8d6de8
	.long	0x3d4f91a4
	.long	0x3ddfa48c
	.long	0xbd2e595f
	.long	0xbce6ed17
	.long	0x3e6363c8
	.long	0x3e05534a
	.long	0xbd581209
	.long	0x3dff40c4
	.long	0x3db0c286
	.long	0x3d4afe49
	.long	0x3e46bb1e
	.long	0xbea7712c
	.long	0xbe523d55
	.long	0xbe73fc40
	.long	0xbe8dc70a
	.long	0x3d79a817
	.long	0x3e47c267
	.long	0x3da3202d
	.long	0xbe176e83
	.long	0xbe4e9f0d
	.long	0xbda0ff2f
	.long	0xbcad0f05
	.long	0x3f1f53d2
	.long	0xbe030a44
	.long	0xbdba4510
	.long	0xbdd81f3e
	.long	0x3e94cdab
	.long	0xbd65f379
	.long	0x3d6d2d5e
	.long	0xbda9b70f
	.long	0x3c9ed723
	.long	0x3eade704
	.long	0xbd2a139f
	.long	0xbe3318f3
	.long	0x3e073d5f
	.long	0xbc6eb47e
	.long	0xbdf1236b
	.long	0x3d8c75f1
	.long	0xbd2ec8d6
	.long	0x3e1129c2
	.long	0x3d826b55
	.long	0xbe451c70
	.long	0xbde844c6
	.long	0xbc46466b
	.long	0x3ddb8ed5
	.long	0x3e9a8f12
	.long	0xbdffa171
	.long	0xbd16bb76
	.long	0x3e960973
	.long	0xbe864900
	.long	0xbd5ba816
	.long	0xbdae099d
	.long	0x3df8dfda
	.long	0xbc008df2
	.long	0xbe978a07
	.long	0x3da72af8
	.long	0x3ec4c703
	.long	0x3db3cd82
	.long	0x3d865927
	.long	0x3e16d0d0
	.long	0x3e0058f7
	.long	0xbd63c7c9
	.long	0x3e80d743
	.long	0xbac91c2e
	.long	0xbe389ead
	.long	0xbe28980d
	.long	0xbdf0c0c3
	.long	0xbded3b04
	.long	0xbddb52cb
	.long	0x3e2a2c7b
	.long	0x3db5da19
	.long	0x3deabc01
	.long	0xbe463f25
	.long	0xbe1b0f1a
	.long	0x3d6554fa
	.long	0x3e0b33b6
	.long	0xbcfd470e
	.long	0xbc4adf9d
	.long	0x3da8b680
	.long	0x3d85701b
	.long	0xbd749ab6
	.long	0xbe18b4b0
	.long	0x3e0402f8
	.long	0x3e7a0be2
	.long	0xbe2b9e94
	.long	0x3db6d666
	.long	0x3d126164
	.long	0xbdb6ecd9
	.long	0xbb3c502f
	.long	0xbe0949ba
	.long	0xbe6c7a97
	.long	0x3d1ffc06
	.long	0xbe2b431b
	.long	0x3e36e8f6
	.long	0xbdc133e5
	.long	0xbc99d1b6
	.long	0x3e0c53ae
	.long	0xbea4ff9a
	.long	0x3e5ed56c
	.long	0x3ea9fa1a
	.long	0xbef384ad
	.long	0x3ee524c7
	.long	0xbe9da15a
	.long	0xbe1c5bc5
	.long	0x3e3a8d41
	.long	0xbdb5fcd3
	.long	0x3e2906e3
	.long	0x3d2a2792
	.long	0x3e3b0741
	.long	0x3b3bceae
	.long	0xbe9bfcbf
	.long	0x3e1a8f49
	.long	0xbd8a92e6
	.long	0xbe2ac692
	.long	0xbe0cfab5
	.long	0x3db2313a
	.long	0xbd25a69d
	.long	0xbd177d2f
	.long	0xbe394446
	.long	0x3d28250d
	.long	0x3dfe9298
	.long	0xbd2da7b9
	.long	0xbd827c6e
	.long	0xbe0ae35d
	.long	0x3db879bc
	.long	0x3e13689e
	.long	0x3e49a1a3
	.long	0xbcf88e8b
	.long	0x3d3b1f26
	.long	0x3d86b3c5
	.long	0xbd417fbf
	.long	0x3db68451
	.long	0x3d2547c6
	.long	0xbd561d4d
	.long	0xbe1f845c
	.long	0x3e44e74c
	.long	0x3e0aae86
	.long	0xbc40814c
	.long	0x3d81464c
	.long	0xbce16516
	.long	0x3ca14e26
	.long	0xbd9c4a22
	.long	0x3d274c2d
	.long	0xbd09ced8
	.long	0xbd8f7622
	.long	0xbd4c468d
	.long	0xbe16f5ef
	.long	0x3d4ec89a
	.long	0xbc77e5c8
	.long	0x3ddd8e15
	.long	0x3dcb2ca7
	.long	0xbe0bdf64
	.long	0xbd25f256
	.long	0xbdba3baa
	.long	0x3e71591d
	.long	0xbe84f49a
	.long	0xbd35d0eb
	.long	0x3cec2f26
	.long	0x3d540316
	.long	0x3b7ee3af
	.long	0x3e20ab28
	.long	0x3dcf717e
	.long	0xbe54655f
	.long	0xbdc09406
	.long	0x3e828549
	.long	0xbe1b3b6d
	.long	0xbec1949e
	.long	0x3e1921eb
	.long	0x3dd2a8c7
	.long	0xbe5f58ed
	.long	0xba65de3b
	.long	0xbdee0ec8
	.long	0x3dba14e8
	.long	0x3f08261b
	.long	0x3e3d425a
	.long	0x3e1e5d62
	.long	0xbd9db202
	.long	0x3d9b0f89
	.long	0x3e75b39b
	.long	0xbee73d99
	.long	0x3b063c79
	.long	0xbecbd58a
	.long	0xbe05cc18
	.long	0xbcccb6e7
	.long	0x3d8d4444
	.long	0xbd2a5df1
	.long	0x3d867b45
	.long	0xbd81f83a
	.long	0x3d3fb7f7
	.long	0xbc4dae88
	.long	0xbded189c
	.long	0xbe6ecaaa
	.long	0xbe8db8fd
	.long	0xbc5f53bb
	.long	0xbe95b692
	.long	0xbecddd14
	.long	0xbebbdff0
	.long	0x3eac41be
	.long	0xbd4e4ed8
	.long	0xbd2b24be
	.long	0xbe851a86
	.long	0x3e997e63
	.long	0x3ee043e6
	.long	0x3d747a36
	.long	0xbc018465
	.long	0xbec35075
	.long	0xbe605201
	.long	0x3e80e759
	.long	0x3e31dd7e
	.long	0xbe32aeb1
	.long	0xbe1e66a1
	.long	0x3e392bef
	.long	0x3eb72ade
	.long	0xbd336786
	.long	0x3ebe2fa2
	.long	0xbee7b302
	.long	0x3e3fc6fd
	.long	0xbc573c7f
	.long	0xbe126cb9
	.long	0xbf135064
	.long	0xbe045427
	.long	0x3e317559
	.long	0x3ec3ec1b
	.long	0x3e1d077c
	.long	0xbe234ccf
	.long	0x3d17f36d
	.long	0xbc8e1514
	.long	0xbd0d6376
	.long	0xbe3307ca
	.long	0x3ddf3d2f
	.long	0x3d999820
	.long	0x3e1bde1e
	.long	0xbd2902d3
	.long	0x3d8ace1c
	.long	0xbe1ebe8b
	.long	0x3e2778af
	.long	0x3e69529d
	.long	0x3d3fe48b
	.long	0xbcf45ad4
	.long	0x3dd4fd1c
	.long	0xbe423c3e
	.long	0x3e0af6d2
	.long	0xbdb8c840
	.long	0x3e86cc1e
	.long	0xbe91bc71
	.long	0x3f25e98f
	.long	0xbe0b6ae8
	.long	0x3d844d4f
	.long	0x3cf68cd9
	.long	0x3e8dda9a
	.long	0xbe635711
	.long	0x3e24e824
	.long	0xbe75956f
	.long	0xbe65cb21
	.long	0xbc69ae65
	.long	0x3ee91321
	.long	0xbc6a171f
	.long	0xbe49c6de
	.long	0xbe8c431f
	.long	0x3e357b16
	.long	0x3d93e909
	.long	0xbdf4afc0
	.long	0xbd9342b2
	.long	0x3ddf50bd
	.long	0xbe33604b
	.long	0xbdb659cc
	.long	0x3e2fd23d
	.long	0xbd73f40a
	.long	0x3de9859b
	.long	0x3e292f98
	.long	0xbeb40599
	.long	0xbd0d3c20
	.long	0xbdf0700f
	.long	0xbe095bfe
	.long	0xbe23f568
	.long	0x3e873a17
	.long	0x3ddf308d
	.long	0xbdb724f2
	.long	0xbe316b9c
	.long	0xbd5e8b35
	.long	0x3d3b8e2f
	.long	0x3da747cf
	.long	0x3dae4fc1
	.long	0x3da44694
	.long	0xbe778d81
	.long	0x3ebe9802
	.long	0xbdcebc06
	.long	0xbe207155
	.long	0x3d508aaa
	.long	0x3e4c9283
	.long	0xbe7ac4f1
	.long	0x3e0a0edf
	.long	0x3b0fc93c
	.long	0xbcff5435
	.long	0x3df88a64
	.long	0xbe01eee8
	.long	0xbe42218a
	.long	0x3e2aac91
	.long	0x3cf99fa3
	.long	0x3e1ec42b
	.long	0xbd5802c8
	.long	0x3cd73745
	.long	0xbe68ea99
	.long	0xbe09991b
	.long	0x3e9c84fd
	.long	0xbdbe6b2c
	.long	0xbe3d3115
	.long	0x3ecc938c
	.long	0xbe6281fc
	.long	0x3e304b77
	.long	0xbcf2c5c6
	.long	0x3de1382f
	.long	0xbdb8f17a
	.long	0xbb61c6fc
	.long	0x3e180369
	.long	0x3d535196
	.long	0x3e2f1c04
	.long	0x3ca70b59
	.long	0xbe8c27e8
	.long	0xbe673cf6
	.long	0xbe1478fc
	.long	0xbd4dd782
	.long	0x3e83c16b
	.long	0x3e37376e
	.long	0xbed5304d
	.long	0x3cc27098
	.long	0xbe30c20e
	.long	0xbf029f2a
	.long	0x3dfccc83
	.long	0xbe9ca15b
	.long	0x3d18f768
	.long	0x3e4542be
	.long	0x3ef20205
	.long	0xbddeb451
	.long	0x3bc7f6dc
	.long	0xbd3e4ae2
	.long	0xbef95d62
	.long	0x3d252980
	.long	0x3dad6c87
	.long	0x3f038537
	.long	0xbd8645aa
	.long	0x3e19ea6e
	.long	0xbd248d77
	.long	0xbe5d3444
	.long	0x3e7ac758
	.long	0xbe0a42f3
	.long	0x3dd655b4
	.long	0x3f3820e6
	.long	0xbe0233ec
	.long	0xbe872b91
	.long	0x3dfb5ae6
	.long	0x3b662875
	.long	0xbf1f45f4
	.long	0x3d82cacc
	.long	0x3cf0d14e
	.long	0x3eac0e11
	.long	0xbcf18af2
	.long	0x3ddf4826
	.long	0x3de7cab8
	.long	0xbe4f3360
	.long	0x3e78d362
	.long	0x3ddf6980
	.long	0xbea57a9a
	.long	0xbe1154c5
	.long	0xbd83ff68
	.long	0x3e4cbbd7
	.long	0x3d08f872
	.long	0xbc41fe5c
	.long	0x3cbdcf6b
	.long	0x3d051842
	.long	0x3dcb1f3a
	.long	0x3c0fe12e
	.long	0xbed96bef
	.long	0x3e0304ed
	.long	0x3eb1573f
	.long	0xbe928e62
	.long	0x3e89740c
	.long	0x3e56acf9
	.long	0xbe43e9d3
	.long	0x3ea93e5b
	.long	0xbe9b645f
	.long	0xbb29fc8a
	.long	0xbe0024a4
	.long	0x3e7355c3
	.long	0x3e7bb693
	.long	0x3e4d1d73
	.long	0xbde61405
	.long	0xbe8147b3
	.long	0xbde83536
	.long	0xbea840d1
	.long	0xbe5a4142
	.long	0xbd9a3377
	.long	0x3ec551e8
	.long	0xbe726979
	.long	0x3ed3ea52
	.long	0xbdf2b24e
	.long	0x3d25e967
	.long	0x3d705b61
	.long	0x3edb5788
	.long	0xbeaa3eae
	.long	0xbc98ade2
	.long	0xbe9969ec
	.long	0xbdb8227e
	.long	0xbdd4c403
	.long	0x3b512e60
	.long	0xbe00b94e
	.long	0xbd1ce8a0
	.long	0xbed17bc7
	.long	0x3cbe6748
	.long	0xbe3733c3
	.long	0x3ea49a10
	.long	0xbe82b1ff
	.long	0x3de34b4c
	.long	0x3dcd43de
	.long	0x3e037cec
	.long	0x3b4a763e
	.long	0x3e037640
	.long	0xbdd30254
	.long	0x3e6b46f2
	.long	0xbe9fed04
	.long	0x3e15e915
	.long	0x3d543100
	.long	0x3e396bd9
	.long	0xbd9c2770
	.long	0x3c83148e
	.long	0xbe7928b9
	.long	0x3e9b7ce8
	.long	0xbeda2d3f
	.long	0x3e8b8410
	.long	0x3d1c2291
	.long	0x3c8146a6
	.long	0x3ea16bb2
	.long	0xbcae8162
	.long	0x3d48d310
	.long	0xbc56ca06
	.long	0x3d8d5d39
	.long	0x3bc4b5d3
	.long	0xbe3aea8f
	.long	0x3db7cfca
	.long	0x3cec843b
	.long	0xbd190413
	.long	0x3dfb5f31
	.long	0x3d7eca84
	.long	0x3ddfe827
	.long	0xbdba67b4
	.long	0x3e2db908
	.long	0xbea56277
	.long	0xbe10d94a
	.long	0x3e83d1e2
	.long	0x3dffa7e3
	.long	0xbd5bfb83
	.long	0x3bde0bf5
	.long	0x3cee3b4a
	.long	0xbc086e11
	.long	0x3e3bcbe2
	.long	0x3c4f045b
	.long	0xbd73e204
	.long	0x3be7da5f
	.long	0x3e254608
	.long	0x3d53a736
	.long	0xbd07a93f
	.long	0xbe164f7a
	.long	0xbd2b3065
	.long	0x3cc75cbb
	.long	0xbd252478
	.long	0xbd936cd6
	.long	0xbd7e6912
	.long	0xbdcae475
	.long	0xbd9ef006
	.long	0x3dffff8e
	.long	0x3d69525c
	.long	0x3db3a505
	.long	0x3e04685f
	.long	0xbb444952
	.long	0xbdd0c838
	.long	0x3b2f8c51
	.long	0xbd8fa240
	.long	0x3dc3b761
	.long	0x3e207627
	.long	0x3dd4eb45
	.long	0x3e132815
	.long	0xbe4b5102
	.long	0x3e4d7605
	.long	0xbe7ea6f7
	.long	0x3c918eb1
	.long	0x3ddb6838
	.long	0x3efc611c
	.long	0xbe898224
	.long	0xbdf8446a
	.long	0xbe020bdf
	.long	0x3d5755a0
	.long	0xbe89eea8
	.long	0xbddc45f3
	.long	0x3e4a590e
	.long	0x3e55785c
	.long	0x3e40f7eb
	.long	0xbdeb99fd
	.long	0xbe9c4f28
	.long	0xbdef4397
	.long	0xbeb2147d
	.long	0x3f054e57
	.long	0xbef1497e
	.long	0xbe6e7019
	.long	0x3e1e56d6
	.long	0x3e8315a5
	.long	0x3e393cef
	.long	0xbeb2c4b1
	.long	0xbdaa836f
	.long	0xbda817e3
	.long	0x3e4f7148
	.long	0x3e2f3fd9
	.long	0xbe34aede
	.long	0xbe31ebee
	.long	0x3e8a6986
	.long	0xbda5d5e3
	.long	0x3d91f6f6
	.long	0xbee6efb5
	.long	0x3d8bf492
	.long	0xbd7df29b
	.long	0x3e890ba0
	.long	0xbd0e51fc
	.long	0x3d38dd3d
	.long	0x3ce81ad8
	.long	0xbd5cd948
	.long	0x3e8b591a
	.long	0x3df6162e
	.long	0xbd567900
	.long	0x3d64a389
	.long	0x3a93cf26
	.long	0xbdc8758e
	.long	0x3db7248f
	.long	0x3e7630ce
	.long	0xbe2aa7df
	.long	0xbe276ede
	.long	0x3e384536
	.long	0xbc11ac17
	.long	0xbce7d638
	.long	0xbe03bb2c
	.long	0x3d1cdf65
	.long	0xbc0309ad
	.long	0x3ed504f0
	.long	0xbe535924
	.long	0xbe03b06d
	.long	0x3e099ad6
	.long	0xbb0603a1
	.long	0xbe8eba95
	.long	0xbe94410c
	.long	0xbdfb3df4
	.long	0xbe12bed9
	.long	0x3df647b7
	.long	0xbca4be0f
	.long	0x3cf31a30
	.long	0xbd0617f0
	.long	0x3daedf24
	.long	0xbbbec3ea
	.long	0x3e8dd241
	.long	0xbdbe4ab9
	.long	0x3d26dd63
	.long	0xbd84d965
	.long	0x3d89f03b
	.long	0xbed08c5e
	.long	0x3e3507e1
	.long	0xbd4d42e7
	.long	0xbee314d0
	.long	0x3e0834d1
	.long	0xbb4d588a
	.long	0xbf11ab71
	.long	0x3eb4dacd
	.long	0xbe1ce272
	.long	0x3d9394e1
	.long	0x3eaea26d
	.long	0xbc805e68
	.long	0x3c9bf3a5
	.long	0xbde1627f
	.long	0xbe0fc650
	.long	0x3da8450b
	.long	0x3ea1741f
	.long	0x3ea2eef9
	.long	0xbb140cb7
	.long	0x3d070ab9
	.long	0xbd96c636
	.long	0xbe6d7a7d
	.long	0x3da9bd8b
	.long	0x3ea937bd
	.long	0xbdefb804
	.long	0xbe0dfb91
	.long	0x3df078ca
	.long	0xbdf2e020
	.long	0x3e79db19
	.long	0x3c2b72ca
	.long	0xbe26f49e
	.long	0xbdcb8c86
	.long	0xbdd10359
	.long	0xbd11b175
	.long	0xbe4bd422
	.long	0x3e3d5c8b
	.long	0x3a2bcf02
	.long	0xbc34e857
	.long	0xbd3e53f3
	.long	0x3e3917e0
	.long	0xbe551b70
	.long	0xbc2f0efe
	.long	0x3de174a6
	.long	0xbea4e10b
	.long	0xbd618408
	.long	0x3e94f2ce
	.long	0x3e4d4b30
	.long	0xbe97bdaa
	.long	0xbd93f494
	.long	0x3e4467e5
	.long	0xbef7bee0
	.long	0x3e9346cf
	.long	0x3e0fecd0
	.long	0x3e8f256d
	.long	0xbd15a397
	.long	0x3da58f24
	.long	0xbe2f0eb3
	.long	0xbdfecbdd
	.long	0xbe8a6e58
	.long	0x3ec72386
	.long	0xbeb360c0
	.long	0xbdcb6785
	.long	0x3ed12e24
	.long	0x3e27e1a6
	.long	0x3e708238
	.long	0xbda89c70
	.long	0xbe6f128b
	.long	0x3e769ace
	.long	0x3d54072f
	.long	0x3e88deff
	.long	0xbdcbf259
	.long	0xbe17ad07
	.long	0xbe22eba5
	.long	0x3d3c2c5e
	.long	0x3df31f3f
	.long	0xbe791f74
	.long	0xbe0fb8cd
	.long	0x3e54a1ac
	.long	0xbe3cfbf6
	.long	0x3e2266e9
	.long	0x3dff2ac7
	.long	0xbcd9d0ce
	.long	0x3d4d50ba
	.long	0x3ca8be63
	.long	0x3d6d29aa
	.long	0xbd83d6ac
	.long	0xbc7f7745
	.long	0x3e182c58
	.long	0xbdd663f3
	.long	0xbe0177db
	.long	0xbe251b26
	.long	0x3ce58a24
	.long	0xbe2e28a3
	.long	0x3e219b44
	.long	0x3df6a4be
	.long	0xbe28e059
	.long	0x3eb9bf61
	.long	0xbe88801c
	.long	0xbe111192
	.long	0x3dc34601
	.long	0x3d81aa3b
	.long	0xbed9eda6
	.long	0xbdbeee6e
	.long	0x3d8ab927
	.long	0x3dd4a516
	.long	0x3eb55c25
	.long	0x3c295a17
	.long	0xbda5a74e
	.long	0xbd4f4a61
	.long	0xbe4dea0a
	.long	0x3caebabc
	.long	0x3e3c6cd2
	.long	0xbe2d6490
	.long	0x3cb67efe
	.long	0xbde0dfae
	.long	0xbd986fd2
	.long	0xbf133658
	.long	0x3eb520a4
	.long	0x3df710b3
	.long	0x3e526871
	.long	0xbd03b034
	.long	0x3e079899
	.long	0xbe722871
	.long	0xbdbe1cc5
	.long	0xbe0490a6
	.long	0x3dde18aa
	.long	0xbef281ec
	.long	0xbd409320
	.long	0xbe7b320b
	.long	0xbdecf36e
	.long	0xbe3bf799
	.long	0x3f97d324
	.long	0xbe5750bc
	.long	0xbe0c7c95
	.long	0xbdad1621
	.long	0x3ed02ef0
	.long	0x3e2cc0f9
	.long	0x3de97b60
	.long	0x3dfcec1c
	.long	0xbeb5e91e
	.long	0x3e4466f8
	.long	0xbe3d768c
	.long	0x3d864fb4
	.long	0x3e4d100f
	.long	0xbda7831c
	.long	0xbe4456a2
	.long	0xbdad05a3
	.long	0xbdaeb7b0
	.long	0xbbb6500d
	.long	0xbe563030
	.long	0x3daf42b9
	.long	0x3b52b040
	.long	0x3d16f270
	.long	0x3d1ad753
	.long	0xbdf8f573
	.long	0x3dd5fe6a
	.long	0x3cc021ca
	.long	0xbde15bd1
	.long	0x3d706160
	.long	0xbd634889
	.long	0xbb4e2079
	.long	0xbdc90250
	.long	0x3d672b93
	.long	0xbdaa51aa
	.long	0xbd91f184
	.long	0xbea9048f
	.long	0x3e58fa2e
	.long	0xbd0de740
	.long	0x3e43fa1f
	.long	0x3e050b8b
	.long	0x3dd99286
	.long	0xbe7284d8
	.long	0x3dde2b11
	.long	0xbe9f6b86
	.long	0x3e2dc3e4
	.long	0xbd9ec722
	.long	0x3f02b096
	.long	0x3df12598
	.long	0x3e3b7067
	.long	0xbe2fa143
	.long	0x3e574cf8
	.long	0xbe98e845
	.long	0x3ec29f23
	.long	0xbc547a4a
	.long	0x3d946aaf
	.long	0xbe892e56
	.long	0x3e4eb1c6
	.long	0xbd415bd2
	.long	0x3dfe20a1
	.long	0xbd87fd4a
	.long	0x3d8abe3e
	.long	0xbd865b49
	.long	0x3d65f96f
	.long	0xbd08b710
	.long	0xbd3d2da4
	.long	0xbde4d073
	.long	0xbbbf554e
	.long	0x3c42c993
	.long	0xbcabebeb
	.long	0x3e8042fc
	.long	0xbde08709
	.long	0xbc03f44e
	.long	0xbe72db63
	.long	0xbb88f5c9
	.long	0x3e57b9ba
	.long	0xbdd2f61f
	.long	0xbdae6913
	.long	0xbd5f397b
	.long	0x3e326c1a
	.long	0x3e86e439
	.long	0xbe20d2c2
	.long	0xbc5be9ea
	.long	0xbe0977cb
	.long	0x3c8cfbeb
	.long	0x3de11c01
	.long	0x3df99a58
	.long	0xbe1da7df
	.long	0x3df41ee0
	.long	0x3d2404ea
	.long	0x3e5af91b
	.long	0xbe36a668
	.long	0x3db2259f
	.long	0xbe7ff338
	.long	0xbd6b4306
	.long	0x3d4512ef
	.long	0xbce8332e
	.long	0x3d1352d9
	.long	0xbd5d4b3b
	.long	0x3d235ef4
	.long	0x3e465180
	.long	0xbdb0cdff
	.long	0x3dc348e1
	.long	0xbda56f9a
	.long	0x3e015b7e
	.long	0xbddf66cc
	.long	0xbe96476f
	.long	0xbd90b859
	.long	0xbde1fb4a
	.long	0x3ea66acf
	.long	0x3de03c08
	.long	0xbd2c32b4
	.long	0x3e3d121a
	.long	0xbdb3fccb
	.long	0xbd51f5bb
	.long	0x3dbbab1c
	.long	0xbd95c738
	.long	0xbddc6b41
	.long	0x3e59d6ec
	.long	0x3e5e9326
	.long	0x3daed63b
	.long	0xbe7776bf
	.long	0x3c7ee480
	.long	0x3e15735b
	.long	0x3e8170ff
	.long	0xbdba3d1d
	.long	0x3e1aa5cc
	.long	0xbd3641ab
	.long	0x3e09a93d
	.long	0xbe02ce01
	.long	0xbe48a926
	.long	0x3ea78034
	.long	0xbebdf042
	.long	0xbeb8a8cb
	.long	0x3e27d5f8
	.long	0xbe4abb06
	.long	0xbe5bf2f2
	.long	0x3e535cfa
	.long	0xbe0476b3
	.long	0xbd450c8a
	.long	0x3df32869
	.long	0xbc1206e1
	.long	0x3d89c88c
	.long	0xbe6fa7e2
	.long	0x3b86ee6b
	.long	0xbeb273d8
	.long	0xbe9d87e1
	.long	0xbe017ed1
	.long	0xbc8e8f0e
	.long	0xbd2b65a3
	.long	0x3ece296a
	.long	0xbe355455
	.long	0x3e877655
	.long	0x3e35cb50
	.long	0x3d1aa7df
	.long	0xbd2f629c
	.long	0xbe6d5777
	.long	0x3d727226
	.long	0xbe1fcab1
	.long	0x3c4810e1
	.long	0x3db7725a
	.long	0xbe34c9ec
	.long	0x3d61c5e6
	.long	0xbe26aac6
	.long	0x3db172be
	.long	0x3ed08a28
	.long	0xbe97b069
	.long	0xbe7fc886
	.long	0x3e151e22
	.long	0x3e0469dd
	.long	0xbed6e4db
	.long	0x3d5f8e80
	.long	0xbe1d6cf4
	.long	0x3e668688
	.long	0x3daa5675
	.long	0xbe264d54
	.long	0xbdb93ff6
	.long	0x3d421a41
	.long	0x3dfee8d5
	.long	0xbcb377bd
	.long	0xbdb9dea8
	.long	0xbe164fe1
	.long	0xbdcacc47
	.long	0x3e3c12fa
	.long	0x3d999829
	.long	0xbe7e8d3b
	.long	0x3e8ac3a8
	.long	0xbd03f157
	.long	0x3e087422
	.long	0xbd77b1c3
	.long	0x3d9bbe53
	.long	0xbe31c91d
	.long	0x3e39161a
	.long	0xbe9c4840
	.long	0x3ddcc44c
	.long	0x3bff6596
	.long	0x3db24f39
	.long	0xbe8f845c
	.long	0xbe125341
	.long	0xbf034ad0
	.long	0x3e66d8ab
	.long	0x3e015234
	.long	0x3e8b19a3
	.long	0xbd2f3d86
	.long	0xbf3f5ee1
	.long	0x3d4651c4
	.long	0x3e2f712a
	.long	0xbe8465fe
	.long	0x3f45cedb
	.long	0xbecc2723
	.long	0xbec3c32f
	.long	0x3f14659f
	.long	0xbe82c413
	.long	0x3f35f2d1
	.long	0xbe80a6ac
	.long	0xbda69bc0
	.long	0x3e022fd8
	.long	0xbd7994ef
	.long	0x3e6ee7a2
	.long	0x3cea0698
	.long	0xbdc14619
	.long	0xbdeb5e59
	.long	0x3d09a58d
	.long	0x3e43b018
	.long	0xbd8de8da
	.long	0xbd4968ae
	.long	0xbd8c4f86
	.long	0xbd5447fb
	.long	0xbe348435
	.long	0xbe3e1963
	.long	0xbdb51b55
	.long	0x3e22e5a4
	.long	0x3d176ec6
	.long	0x3da906d9
	.long	0x3d3691a9
	.long	0xbcd779e2
	.long	0xbd90bd5b
	.long	0x3ddd80bc
	.long	0x3c061f17
	.long	0xbe301479
	.long	0x3e24a9ba
	.long	0xbda2caf4
	.long	0x3d563f22
	.long	0x3e69af77
	.long	0xbe9c6cc6
	.long	0x3ddcce53
	.long	0x3d60e735
	.long	0x3ed23add
	.long	0x3c51c915
	.long	0xbe3d6421
	.long	0x3dce813f
	.long	0x3cf3f728
	.long	0xbe6476a1
	.long	0xbdda9ca4
	.long	0xbe92adb5
	.long	0x3b32cce7
	.long	0xbc99add1
	.long	0x3ce45fa9
	.long	0xbce7196c
	.long	0x3e35433c
	.long	0xba952b39
	.long	0xbe2d6587
	.long	0x3d3af2e5
	.long	0xbe014172
	.long	0xbdc6ce6a
	.long	0xbddaed7b
	.long	0x3d82779c
	.long	0xbd499174
	.long	0xbe08bd85
	.long	0xbe02a407
	.long	0x3cc44380
	.long	0x3cb241d6
	.long	0x3d9f854f
	.long	0xbe2a5ec5
	.long	0x3e655683
	.long	0x3e405872
	.long	0xbdf8437d
	.long	0x3d751779
	.long	0xbe0105d1
	.long	0xbe6efc93
	.long	0xbd95f678
	.long	0x3a825337
	.long	0xbc790bb9
	.long	0xbe4b42f7
	.long	0xbe41b9a4
	.long	0x3e0b755a
	.long	0x3e58a6d8
	.long	0xbcb4baab
	.long	0xbc899f11
	.long	0x3e0af264
	.long	0x3dcd8fe9
	.long	0x3ee8d0ab
	.long	0xbe27a5d2
	.long	0xbe6a5a1a
	.long	0x3c8f5c52
	.long	0xbe5a8489
	.long	0xbdc7b2e4
	.long	0x3e4ed2f8
	.long	0x3e85a244
	.long	0x3e5c71c4
	.long	0x3e13f369
	.long	0x3bafde37
	.long	0x3dfa68c1
	.long	0xbdfe8e4c
	.long	0x3c82f4bc
	.long	0xbe02d6b2
	.long	0x3dae3c51
	.long	0xbe654ade
	.long	0x3dae479d
	.long	0xbd376b33
	.long	0xbd28c598
	.long	0xbc0c2196
	.long	0xbddc934a
	.long	0xbde9ef8f
	.long	0xbe22c803
	.long	0x3e643290
	.long	0xbdecab40
	.long	0xbe30f489
	.long	0xbc995682
	.long	0xbc82833a
	.long	0x3c4e004e
	.long	0x3cc82b16
	.long	0x3e17f97a
	.long	0x3e2ab909
	.long	0xbe587d92
	.long	0x3c70b813
	.long	0xbe4671bb
	.long	0xbe1c5d5e
	.long	0x3e2eb89d
	.long	0x3d9905dd
	.long	0xbdf3bf76
	.long	0xbdcb3382
	.long	0x3d158467
	.long	0x3e9f4317
	.long	0xbdc47f3c
	.long	0xbdd1089d
	.long	0x3ee61287
	.long	0xbe465cb1
	.long	0x3e7480c0
	.long	0xbe042e68
	.long	0x3cdc0bbf
	.long	0x3e8362e6
	.long	0xbe4b8537
	.long	0xbe03ad90
	.long	0xbd01136b
	.long	0xbdd9ae33
	.long	0xbd1c0056
	.long	0x3d034b37
	.long	0xbd7f6b39
	.long	0xbda51631
	.long	0x3e08611c
	.long	0x3da97661
	.long	0x3da85490
	.long	0x3cfed78f
	.long	0x3c2f7f48
	.long	0x3cf77bb2
	.long	0xbeec7eba
	.long	0x3e3c1188
	.long	0x3cb16620
	.long	0xbda3fbc3
	.long	0x3c8b09fb
	.long	0xbd35488e
	.long	0x3d9b4c28
	.long	0x3e0dbfa3
	.long	0x3e6fbf10
	.long	0xbca35d4c
	.long	0x3d89e293
	.long	0xbd8ed571
	.long	0x3d4e7a92
	.long	0x3d1bdbfd
	.long	0xbdfa3335
	.long	0xbdb28ebc
	.long	0xbd828173
	.long	0x3e2dd0ea
	.long	0xbde22f40
	.long	0x3d71d7e4
	.long	0x3eeb001a
	.long	0xbe136b87
	.long	0x3e1bc4d1
	.long	0x3e142ac9
	.long	0xbd9c57c6
	.long	0xbe9a2564
	.long	0xbd587e00
	.long	0x3e6c11bc
	.long	0xbe6594d0
	.long	0xbdde444d
	.long	0xbd41a90b
	.long	0x3e0d4bc1
	.long	0xbe81324e
	.long	0x3e1d9778
	.long	0xbda7199c
	.long	0x3e36c9ee
	.long	0x3e7199ca
	.long	0xbe388463
	.long	0x3d66d518
	.long	0x3e3a9d66
	.long	0xbd78b567
	.long	0x3d10c4ae
	.long	0xbdabb4c0
	.long	0xbc200217
	.long	0xbdd8b919
	.long	0xbdce26e1
	.long	0x3d973314
	.long	0x3e26650b
	.long	0x3e2671df
	.long	0x3d94ce05
	.long	0xbe88f06b
	.long	0x3d8937e9
	.long	0x3e485589
	.long	0x3e32235d
	.long	0xbeaa07c1
	.long	0xbe597be8
	.long	0x3cb00bf9
	.long	0xbd25aece
	.long	0x3da9ae30
	.long	0xbd4b721d
	.long	0xbd2d36e7
	.long	0xbd5b975f
	.long	0x3e704fc8
	.long	0x3e8d03e4
	.long	0x3d19487e
	.long	0x3e73a5cb
	.long	0x3c709d01
	.long	0xbdd09cd7
	.long	0x3dad81de
	.long	0xbd51878c
	.long	0x3cd0d387
	.long	0xbeccc5f2
	.long	0x3eb0153f
	.long	0xbd982912
	.long	0x3dc935a3
	.long	0x3e1da937
	.long	0xbe6b41ce
	.long	0x3e3dcfe6
	.long	0x3d629cc6
	.long	0xbce214e4
	.long	0xbdc0a653
	.long	0xbde259f4
	.long	0xbdaa84e5
	.long	0x3e2991ad
	.long	0x3e457278
	.long	0xbe25f0c3
	.long	0xbe91b851
	.long	0xbb637355
	.long	0xbddffc76
	.long	0x3e87e810
	.long	0xbd07c189
	.long	0xbdbf1dbf
	.long	0xbe0e2bde
	.long	0x3d912884
	.long	0x3e850cd1
	.long	0x3e3071db
	.long	0xbe8292a7
	.long	0xbd683ab3
	.long	0x3d95ce37
	.long	0xbd755f23
	.long	0xbdbc845b
	.long	0xbe66d7d5
	.long	0xbe1e28a0
	.long	0xba990fd4
	.long	0x3e88a82a
	.long	0x3e011787
	.long	0x3ea4fc3c
	.long	0xbe047b06
	.long	0x3d282d5a
	.long	0xbe1c8c2b
	.long	0x3eef66d4
	.long	0xbe804b8d
	.long	0xbda56da6
	.long	0xbe180dbc
	.long	0x3d974e80
	.long	0xbea6c6d4
	.long	0xbd8a5715
	.long	0x3bfc865a
	.long	0xbc80ce81
	.long	0x3e721b06
	.long	0x3eed1bd7
	.long	0xbe8ec62c
	.long	0xbe18d870
	.long	0xbd9d367b
	.long	0x3e7151c4
	.long	0xbd532a74
	.long	0x3f00933a
	.long	0xbf054b8b
	.long	0x3c181b48
	.long	0xbc30b933
	.long	0x3ec4941a
	.long	0xbe21ae0f
	.long	0x3cdb4d51
	.long	0xbe80699f
	.long	0x3e08ed5f
	.long	0x3c67581f
	.long	0xbe699643
	.long	0x3ec2fcc0
	.long	0x3da8bbf5
	.long	0xbd4e3eee
	.long	0xbe8f252d
	.long	0x3e19946e
	.long	0xbe041b58
	.long	0x3dd0c9b4
	.long	0xbdb6b083
	.long	0xbda3e212
	.long	0x3be12260
	.long	0xbe145df5
	.long	0xbdf23cb0
	.long	0x3db100e3
	.long	0xbdfca4d3
	.long	0x3ec7ea09
	.long	0x3b498735
	.long	0xbd4bb9c2
	.long	0xbe600e66
	.long	0x3e1cc9bb
	.long	0x3e091999
	.long	0x3dec7b6d
	.long	0xbd038e7b
	.long	0xbdd71ac2
	.long	0x3e56b7f6
	.long	0xbcec2861
	.long	0xbe67a1aa
	.long	0x3dc630e6
	.long	0xbe61c134
	.long	0x3dfbdaae
	.long	0xbe0e7014
	.long	0x3d30075c
	.long	0xbdcaa947
	.long	0x3d85b093
	.long	0x3cf005e3
	.long	0x3dafd94a
	.long	0xbe4a0f6f
	.long	0x3dc88fa1
	.long	0xbcddf3dd
	.long	0xbe18707d
	.long	0xbccf91cc
	.long	0x3e74da60
	.long	0xbc5d7b59
	.long	0xbdcc2a45
	.long	0xbdbab2b3
	.long	0x3c83b382
	.long	0x3e8df36e
	.long	0xbdd74c47
	.long	0xbdc46d2a
	.long	0x3d2d867d
	.long	0xbd4f2f83
	.long	0xbe59c3aa
	.long	0x3df11f36
	.long	0x3d595f71
	.long	0xbe177a05
	.long	0xbcdb4d23
	.long	0x3dbf3912
	.long	0xbdcfdbfb
	.long	0xbcc68926
	.long	0x3d4dd304
	.long	0xbc1e3922
	.long	0x3dd90132
	.long	0x3da4a411
	.long	0xbe19201f
	.long	0x3d93a986
	.long	0x3da8620c
	.long	0xbdbe9a2d
	.long	0x3dbefaa7
	.long	0xbe8f0ae6
	.long	0xbc202b9a
	.long	0x3e267b6d
	.long	0x3d3a1985
	.long	0xbb020845
	.long	0xbd8d1102
	.long	0xbe3e70cd
	.long	0x3b96a23b
	.long	0xbe53fce9
	.long	0x3d68b2ec
	.long	0x3e26e640
	.long	0xbe2168ba
	.long	0xbdcd04e1
	.long	0x3d8c0153
	.long	0xbdabacfa
	.long	0x3d7e0700
	.long	0xbe858a10
	.long	0xbe3b3710
	.long	0xbe1e5adb
	.long	0x3e62b9dd
	.long	0x3e986b44
	.long	0x3e117750
	.long	0x3cdd262a
	.long	0xbe410f27
	.long	0xbdd5564f
	.long	0x3de520e9
	.long	0x3e453657
	.long	0x3df7dffe
	.long	0x3b7e658a
	.long	0xbd2b3e01
	.long	0xbe03322f
	.long	0x3bcfc079
	.long	0xbdfe81e8
	.long	0xbd5c3919
	.long	0xbd73b927
	.long	0x3d0abcdc
	.long	0x3e8c2132
	.long	0xbbdd8725
	.long	0xbdc901ed
	.long	0xbcfb015f
	.long	0x3dcf735e
	.long	0x3d8d7d33
	.long	0xbcae604a
	.long	0xbe53ce2d
	.long	0x3e2428bf
	.long	0x3f013feb
	.long	0x3dffcf20
	.long	0xbdebf37b
	.long	0xbe66f795
	.long	0xbc6bec1e
	.long	0x3dbaa656
	.long	0xbe1ef8f2
	.long	0xbe563057
	.long	0xbdc34a14
	.long	0xbea1ab78
	.long	0x3e219c47
	.long	0xbe0ce048
	.long	0x3de20765
	.long	0x3a49317d
	.long	0x3e337686
	.long	0x3e932870
	.long	0x3e253f15
	.long	0xbe3f825d
	.long	0xbdae4615
	.long	0xbe829169
	.long	0x3dcdcdbf
	.long	0xbdb5d676
	.long	0x3c87b0b6
	.long	0xbd855f78
	.long	0x3e0c7009
	.long	0xbe813b29
	.long	0x3e77d5e8
	.long	0x3e5485e6
	.long	0xbd330fc9
	.long	0x3e84635e
	.long	0x3e35adce
	.long	0x3e09e41b
	.long	0xbea389e6
	.long	0xbccc4311
	.long	0xbe7d4f27
	.long	0xbe69e827
	.long	0x3e964d0d
	.long	0xbe222a0b
	.long	0xbb943292
	.long	0x3d93814e
	.long	0x3ea4b332
	.long	0x3d920ecc
	.long	0xbb168bbf
	.long	0xbddb2cc9
	.long	0x3e2cc0ed
	.long	0xba75ef86
	.long	0x3e8a78fb
	.long	0xbddb03e4
	.long	0xbe89fa38
	.long	0xbdbf485a
	.long	0xbeb53305
	.long	0xbe9a172a
	.long	0x3eb45585
	.long	0xbd87d61e
	.long	0xbea294e2
	.long	0xbe6933dd
	.long	0xbd622bb5
	.long	0x3ef3e08d
	.long	0x3ea14a51
	.long	0x3befbdf3
	.long	0xbec592d4
	.long	0xbdce3a43
	.long	0x3e190d4c
	.long	0x3cfec926
	.long	0x3daee41e
	.long	0x3db7f76a
	.long	0xbccf7d19
	.long	0xbdc8add9
	.long	0x3e85aace
	.long	0x3e49d30c
	.long	0xbbfb3e70
	.long	0x3efcfb3f
	.long	0xbe3b8744
	.long	0x3de3ff24
	.long	0xbdb8e33e
	.long	0x3e2ba942
	.long	0xbe6edc20
	.long	0xbe600607
	.long	0xbe4aa978
	.long	0x3df920e6
	.long	0x3d8b04f4
	.long	0x3e142149
	.long	0xbe615adf
	.long	0xbd3e621f
	.long	0xbca98fd1
	.long	0x3aa7a1cd
	.long	0x3e118a85
	.long	0xbe80755d
	.long	0x3c7d0429
	.long	0x3eb0bccf
	.long	0xbeb15692
	.long	0x3cab30fc
	.long	0x3e9df1c6
	.long	0x3d703355
	.long	0x3d3ed000
	.long	0x3e62b438
	.long	0xbdfe162e
	.long	0xbd9f23b7
	.long	0xbd992196
	.long	0xbddbe798
	.long	0xbd98841f
	.long	0xbd1aa29f
	.long	0x3d83097f
	.long	0x3de83f92
	.long	0xbe0cd2a3
	.long	0x3e2a742e
	.long	0x3d0558e6
	.long	0x3ddc6eb9
	.long	0xbd27465f
	.long	0xbb22115e
	.long	0x3e9d1be2
	.long	0xbe2c6317
	.long	0xbe916cc6
	.long	0x3eb6f5e6
	.long	0x3e1f2ced
	.long	0x3e3afe02
	.long	0x3e15b0b8
	.long	0xbe38a3ef
	.long	0xbe79e140
	.long	0xbd89fe12
	.long	0x3e15edcf
	.long	0xbd001505
	.long	0xbec1ffe2
	.long	0x3ebabc71
	.long	0x3db4aa6a
	.long	0xbd949680
	.long	0xbd8fc417
	.long	0x3bfe5a0e
	.long	0xbedb943d
	.long	0xbdbc713e
	.long	0x3ec88d2c
	.long	0xbe9f872d
	.long	0xbdeb7156
	.long	0x3e751f12
	.long	0x3da71fe8
	.long	0x3dbce9c1
	.long	0xbda45e7c
	.long	0xbd8af5f7
	.long	0xbea498f8
	.long	0x3e588dfd
	.long	0xbe089de5
	.long	0xbe70b5e4
	.long	0xbe37875c
	.long	0x3d7958d5
	.long	0xbd2faf81
	.long	0x3e5e5a93
	.long	0xbe5337a9
	.long	0x3e4de247
	.long	0x3da56258
	.long	0x3e657f05
	.long	0xbf14226e
	.long	0xbd45006f
	.long	0xbe726429
	.long	0x3eacab46
	.long	0x3e121083
	.long	0x3e980aaa
	.long	0xbe89feb0
	.long	0x3d4ffd00
	.long	0xbc752d7c
	.long	0xbe0ae1ae
	.long	0x3c8067b9
	.long	0x3c665d78
	.long	0x3f0a4f27
	.long	0xbed9c577
	.long	0xbe84bd3a
	.long	0x3d6d47ff
	.long	0x3e86fe27
	.long	0x3da49a8a
	.long	0xbe78a502
	.long	0x3df70375
	.long	0x3db81be2
	.long	0x3efa8729
	.long	0x3e37735c
	.long	0xbd77a6fc
	.long	0xbf2eb276
	.long	0xbe86cb4e
	.long	0xbe0d86f4
	.long	0x3e8f6965
	.long	0x3e0b4cc5
	.long	0x3e8ea54e
	.long	0xbe122ebe
	.long	0xbeba9db4
	.long	0x3c2ed19c
	.long	0xbd3e3e56
	.long	0xbde5baee
	.long	0x3e6c24b0
	.long	0xbd8fda85
	.long	0x3e80a01e
	.long	0x3df30278
	.long	0x3e9436ff
	.long	0xbea0a902
	.long	0x3de65309
	.long	0xbf04356b
	.long	0x3eb70511
	.long	0xbe45d4fc
	.long	0xbe34d678
	.long	0xbd443ea1
	.long	0x3e1efd1e
	.long	0x3ebbb8b8
	.long	0xbdc9a2cc
	.long	0xbd3f86f0
	.long	0x3e48a9c7
	.long	0xbe08b2bf
	.long	0xbd5f4a57
	.long	0xbc0c4fbc
	.long	0xbe51ac88
	.long	0xbe51e577
	.long	0xbe317d46
	.long	0x3d086ba1
	.long	0xbd70b501
	.long	0xbb9f157e
	.long	0xbe424bc6
	.long	0x3df33675
	.long	0xbdba161b
	.long	0x3dc10ef9
	.long	0x3e10da28
	.long	0x3e6222a6
	.long	0x3ce24b18
	.long	0xbd662ede
	.long	0x3e226fc3
	.long	0xbe13dd48
	.long	0x3e286447
	.long	0xbe1ca6a2
	.long	0xbe2b1005
	.long	0x3d2ba0c2
	.long	0xbdbc61d6
	.long	0x3dc3fe6c
	.long	0xbcd7c908
	.long	0xbd34e00c
	.long	0xbe328d1e
	.long	0x3e9181db
	.long	0xbef881a3
	.long	0xbe969f34
	.long	0x3e866b49
	.long	0xbdb48a0e
	.long	0xbe5a74ad
	.long	0x3ee2c5aa
	.long	0xbdce0a52
	.long	0x3ec9ef15
	.long	0xbf0d164e
	.long	0x3d74061a
	.long	0xbe1cfc85
	.long	0x3e08a75e
	.long	0x3ef22fa7
	.long	0x3d3b15a7
	.long	0xbe423603
	.long	0x3f0be23b
	.long	0xbeb2ccae
	.long	0x3f00be64
	.long	0xbe881f07
	.long	0xbda273a8
	.long	0xbec52aff
	.long	0x3e9cd74e
	.long	0x3e2aefbe
	.long	0x3dca7e83
	.long	0xbc16a9df
	.long	0x3d64d08f
	.long	0x3e66d8c5
	.long	0x3d106cdb
	.long	0xbd5f849b
	.long	0xbba22e44
	.long	0xbe078d09
	.long	0xbdd0e3ee
	.long	0x3cfee3e1
	.long	0xbcf836f1
	.long	0xbdad1536
	.long	0xbd1b6276
	.long	0xbdd4d285
	.long	0xbdc644bf
	.long	0xbdcf983b
	.long	0x3ea5bf3f
	.long	0xbe898938
	.long	0x3d4bcf84
	.long	0x3dd99e56
	.long	0x3baa1d3f
	.long	0xbe25def2
	.long	0x3dce32af
	.long	0xbd5d659d
	.long	0xbe5766b3
	.long	0x3df4c1fe
	.long	0x3e860729
	.long	0xbe0b0d18
	.long	0x3c02f059
	.long	0x3ebaa292
	.long	0xbe94c349
	.long	0x3e1ad936
	.long	0x3cfd2e29
	.long	0x3dba4715
	.long	0xbeecf962
	.long	0xbe5ef69a
	.long	0xbba736b7
	.long	0x3e151d46
	.long	0x3e5b9006
	.long	0x3e52f9dc
	.long	0xbe109ec4
	.long	0x3e21f306
	.long	0xbc0ecd22
	.long	0xbdc8ebbf
	.long	0x3da2c201
	.long	0xbe0f03af
	.long	0x3d7cda31
	.long	0x3d896ca8
	.long	0xbc8a4f7d
	.long	0x3dca67cc
	.long	0xbd64e351
	.long	0x3c7b9174
	.long	0x3e0ae7de
	.long	0xbe00190e
	.long	0xbd04c533
	.long	0x3d8b8f4c
	.long	0x3d531d01
	.long	0xbdb771e6
	.long	0xbe461536
	.long	0x3d24671c
	.long	0x3e01270c
	.long	0xbcab5267
	.long	0x3d1ce0ba
	.long	0xbe896a11
	.long	0xbe2bb983
	.long	0x3e08641f
	.long	0x3e47f066
	.long	0xbcddd51a
	.long	0xbc606c54
	.long	0x3e18ab30
	.long	0xbde1c9a7
	.long	0xbe890324
	.long	0x3e2ea4b3
	.long	0xbe9ee4c4
	.long	0xbe057066
	.long	0x3d49ee98
	.long	0xbe22ed82
	.long	0xbc7ff32c
	.long	0x3d51cd0b
	.long	0x3e0f237e
	.long	0xbdd7c319
	.long	0xbe1758b4
	.long	0xbe2cccdc
	.long	0xbd9d133b
	.long	0x3e804252
	.long	0xbb44d40e
	.long	0xbe364371
	.long	0x3c485c95
	.long	0xbd6b8594
	.long	0xbde21c3a
	.long	0xbcd28ef5
	.long	0x3d939333
	.long	0x3e03e88d
	.long	0xbddc5e81
	.long	0xbd1ab7ed
	.long	0xbe1197aa
	.long	0xbc6af168
	.long	0xbde60f0d
	.long	0x3ebec024
	.long	0xbe55fd93
	.long	0x3e0ea774
	.long	0xbddee3dc
	.long	0x3e5369b0
	.long	0xbe0bbed7
	.long	0xbca4b509
	.long	0x3d5c592f
	.long	0x3deb3de1
	.long	0xbd73096b
	.long	0xbe92c0e1
	.long	0xbe4a0ac3
	.long	0x3e84ba1e
	.long	0xbd38385a
	.long	0x3e041383
	.long	0xbd07400f
	.long	0x3dd50c6d
	.long	0xbdd6ed66
	.long	0xbeb90d75
	.long	0xbed0d4a4
	.long	0x3f086a4e
	.long	0xbee21d59
	.long	0x3ef23962
	.long	0x3e6cbaaf
	.long	0xbeac3b34
	.long	0x3ecf13c5
	.long	0xbe21ed9a
	.long	0xbdc721d6
	.long	0xbd67c6a3
	.long	0xbdacc148
	.long	0xbd925019
	.long	0x3e1fef57
	.long	0x3e39e34a
	.long	0xbe2b2b5d
	.long	0xbdc1aafe
	.long	0xbdf763e7
	.long	0x3c9a50cc
	.long	0x3e268af7
	.long	0xbd8aefb4
	.long	0x3d6fed1f
	.long	0xbd304d5f
	.long	0x3e11eb96
	.long	0xbe260144
	.long	0xbe3e1856
	.long	0x3d74bd74
	.long	0x3e892029
	.long	0xbce3b338
	.long	0xbe06b119
	.long	0x3e33558d
	.long	0xbe1a537c
	.long	0x3e05aa3c
	.long	0xbe2eae9f
	.long	0x3d9f9a07
	.long	0xbe9208cf
	.long	0x3dfa449a
	.long	0xbdbdb3e8
	.long	0x3e0c81ac
	.long	0xbe4fdbe1
	.long	0x3b663915
	.long	0x3e4088a1
	.long	0x3ce963b6
	.long	0xbc1a744f
	.long	0x3cb9f432
	.long	0xbdc94662
	.long	0xbe38e7f0
	.long	0x3eef79b2
	.long	0xbe2651c8
	.long	0x3dce30ff
	.long	0xbca01f6b
	.long	0xbd913d82
	.long	0xbd2c8261
	.long	0x3bf230ac
	.long	0xbd588fbc
	.long	0xbded3801
	.long	0xbe014304
	.long	0x3d757919
	.long	0xbe12a84b
	.long	0xbdd5efad
	.long	0x3d290d9e
	.long	0x3defeb18
	.long	0xbdf5e87f
	.long	0x3e344073
	.long	0xbe3383a6
	.long	0x3e83f622
	.long	0x3e2c29c8
	.long	0xbc9d3995
	.long	0xbe44f9fb
	.long	0xbe174e19
	.long	0x3e170827
	.long	0xbe7267c3
	.long	0x3ddd96d3
	.long	0x3d58469a
	.long	0xbe81f541
	.long	0x3ec9de7c
	.long	0xbbf3737b
	.long	0xbdf74400
	.long	0xbe0e0453
	.long	0xbdc7f2f6
	.long	0x3d470400
	.long	0xbe304d3a
	.long	0x3d5bdae0
	.long	0x3d89280d
	.long	0x3cb59a4e
	.long	0xb99ff537
	.long	0xbd034437
	.long	0xbd740604
	.long	0x3d88825b
	.long	0x3d1b259e
	.long	0xbd938412
	.long	0xbc6db2fb
	.long	0xbdcab0f6
	.long	0xbd6e771e
	.long	0xbd8e7aef
	.long	0xbd7fe563
	.long	0xbde04424
	.long	0x3e0bcbb3
	.long	0x3dd9e91a
	.long	0xbe2e0b78
	.long	0xbe44c950
	.long	0xbe9f4b72
	.long	0xbd889b61
	.long	0x3e894a4e
	.long	0xbe3e9495
	.long	0x3ecb9aa5
	.long	0x3cbf1194
	.long	0x3d6c3fa4
	.long	0x3e5e7d91
	.long	0x3e859760
	.long	0x3d886041
	.long	0x3ba637be
	.long	0xbf02faf3
	.long	0xbd146ca9
	.long	0x3e0548ce
	.long	0x3e8e5d79
	.long	0x3d0437da
	.long	0xbebf0ee0
	.long	0x3c4e2e29
	.long	0xbe9428b5
	.long	0x3e875517
	.long	0xbe43f5ba
	.long	0x3d833bd8
	.long	0x3edfd4a1
	.long	0xbec08467
	.long	0xbdf18f49
	.long	0x3e8af5ad
	.long	0x3da08ffa
	.long	0xbdd977b0
	.long	0x3d6a9584
	.long	0xbccd92f9
	.long	0xbd055636
	.long	0xbdc9a253
	.long	0xbd2fa5f7
	.long	0x3d8910d1
	.long	0x3c9df199
	.long	0x3e04904a
	.long	0xbe069e1b
	.long	0x3dd00717
	.long	0x3eca24ef
	.long	0xbe6116ed
	.long	0xbea9434f
	.long	0xbe84d8a3
	.long	0xbcd2e014
	.long	0xbe402f8f
	.long	0x3d249a69
	.long	0x3f0484a9
	.long	0xbd044381
	.long	0x3d079577
	.long	0xbd87cff9
	.long	0x3ebd3679
	.long	0x3db0ef82
	.long	0xbe5e9d01
	.long	0xbea7b75b
	.long	0x3e504274
	.long	0x3d2a8f8a
	.long	0xbd18d4c4
	.long	0x3e83471c
	.long	0xbe85a882
	.long	0x3e985603
	.long	0xbe07be70
	.long	0x3eb2890e
	.long	0xbeccd578
	.long	0xbcc36b65
	.long	0x3e1c8d25
	.long	0xbeb382f0
	.long	0x3ecd9105
	.long	0xbbb93a26
	.long	0xbe82fb9e
	.long	0xbcfbf514
	.long	0xbdf88439
	.long	0xbe10d4eb
	.long	0x3d7214b4
	.long	0xbdb29970
	.long	0x3db98f40
	.long	0x3c14d535
	.long	0x3bd50d2f
	.long	0x3bc2898c
	.long	0xbc08dcbc
	.long	0xbe3622dc
	.long	0x3ee66690
	.long	0xbe1efdb9
	.long	0xbe57c9e7
	.long	0x3d886067
	.long	0x3e263f75
	.long	0xbd8542e6
	.long	0x3ab6e355
	.long	0xbe3d46f7
	.long	0xbde6fcf2
	.long	0xbdcdf73d
	.long	0x3dc90458
	.long	0xbe4d216a
	.long	0x3bb61e27
	.long	0x3eba4e33
	.long	0x3de66fa7
	.long	0xbeed5cb9
	.long	0x3d7b9c27
	.long	0xbebf5527
	.long	0x3f04d37f
	.long	0x3e03e0fa
	.long	0xbdf21477
	.long	0xbe0e582c
	.long	0x3e54ce9f
	.long	0xbe8b4d5f
	.long	0x3e8a2e81
	.long	0xbe632ca8
	.long	0x3e70be82
	.long	0xbd15c313
	.long	0x3df9e79e
	.size	.L__constant_16x4x4x10xf32, 10240

	.type	.L__constant_16x1x1xf32,@object
	.p2align	6, 0x0
.L__constant_16x1x1xf32:
	.long	0xbda87212
	.long	0xbddef698
	.long	0xbe106caf
	.long	0xbe51c939
	.long	0xbe376f56
	.long	0xbe5c9be1
	.long	0xbe090432
	.long	0xbe486c05
	.long	0xbe895826
	.long	0xbe843464
	.long	0xbd9bf7b6
	.long	0x3c59a5bf
	.long	0xbb91a461
	.long	0xbed458e9
	.long	0xbe371507
	.long	0xbd1e55a3
	.size	.L__constant_16x1x1xf32, 64

	.type	.L__constant_2xi64_0,@object
	.p2align	6, 0x0
.L__constant_2xi64_0:
	.quad	1
	.quad	256
	.size	.L__constant_2xi64_0, 16

	.type	.L__constant_1x10xf32,@object
	.p2align	6, 0x0
.L__constant_1x10xf32:
	.long	0xbd37baf4
	.long	0x3bff5131
	.long	0x3d8b7871
	.long	0x3cf5b56f
	.long	0xbe017187
	.long	0x3e0f9581
	.long	0xbd62726c
	.long	0xbd4a46af
	.long	0x3dacb10a
	.long	0xbd5f65c5
	.size	.L__constant_1x10xf32, 40

	.type	.L__constant_2xi64,@object
	.p2align	6, 0x0
.L__constant_2xi64:
	.quad	256
	.quad	10
	.size	.L__constant_2xi64, 16

	.type	.L__constant_8x1x1xf32,@object
	.p2align	6, 0x0
.L__constant_8x1x1xf32:
	.long	0xbe256aab
	.long	0xbede1fb5
	.long	0x3dbbae77
	.long	0xbc8a0da9
	.long	0xbd852c95
	.long	0xbe06e64b
	.long	0x3ca742b5
	.long	0xbdf808a4
	.size	.L__constant_8x1x1xf32, 32

	.type	.L__constant_8x1x5x5xf32,@object
	.p2align	6, 0x0
.L__constant_8x1x5x5xf32:
	.long	0xbc11e916
	.long	0xbe7297dd
	.long	0xbf024223
	.long	0xbd8438f7
	.long	0x3e113720
	.long	0xbf178bc0
	.long	0xbef3589b
	.long	0xbd4a2148
	.long	0x3f44a9c7
	.long	0x3e86e4e9
	.long	0xbefbc86b
	.long	0x3d63cf56
	.long	0x3f826d6e
	.long	0x3f0e0118
	.long	0xbee221d6
	.long	0xbe235daa
	.long	0x3f0ebb0a
	.long	0x3f17934b
	.long	0xbe96e858
	.long	0xbf1cf95a
	.long	0x3d1db0f8
	.long	0x3e67719b
	.long	0xbe5fcd39
	.long	0xbef452d0
	.long	0xbe955a5a
	.long	0xbd235bac
	.long	0x3e5fd8a1
	.long	0x3eff5927
	.long	0x3ed8eb82
	.long	0x3d46da0c
	.long	0xbdcf385e
	.long	0xbe8dc714
	.long	0xbb9025b5
	.long	0x3f06f4c2
	.long	0x3ece8bb3
	.long	0xbe1a7540
	.long	0xbea744e7
	.long	0xbe22cd28
	.long	0x3eb7ce4e
	.long	0x3ebeca17
	.long	0xbf115342
	.long	0xbec57ca0
	.long	0xbe31dd44
	.long	0x3e69829f
	.long	0x3eaa5e14
	.long	0xbec2f5d5
	.long	0xbe6afcb3
	.long	0xbd87a8af
	.long	0xbc6e9042
	.long	0x3e91dfff
	.long	0xbcc2078c
	.long	0x3dd606e6
	.long	0x3e81db6d
	.long	0x3eb93a1f
	.long	0x3f30e2eb
	.long	0x3eab7a2e
	.long	0x3ed580fc
	.long	0x3ec5083c
	.long	0x3ea8a870
	.long	0x3e78d8e9
	.long	0xbef02a09
	.long	0xbd9b9055
	.long	0xbe13fe5b
	.long	0xbe31ba65
	.long	0xbe7cb170
	.long	0xbf7901a3
	.long	0xbf3732b3
	.long	0xbf09c84f
	.long	0xbf1de549
	.long	0xbee59b70
	.long	0xbe4bc226
	.long	0xbeaaad9f
	.long	0xbeb2ba29
	.long	0xbe158a67
	.long	0xbd823f8a
	.long	0xbe973825
	.long	0xbeaa3264
	.long	0xbea7eee8
	.long	0xbda237bd
	.long	0xbe63c4f9
	.long	0xbef41a43
	.long	0xbe57c5fd
	.long	0xbd6ba394
	.long	0x3e27ab2a
	.long	0x3e25dc9b
	.long	0xbec8904c
	.long	0xbcc0a05d
	.long	0x3dfc7307
	.long	0x3d2daba4
	.long	0x3e6985b2
	.long	0x3dc19e72
	.long	0x3ebe7ba6
	.long	0x3e60a70e
	.long	0xbc37816d
	.long	0x3e39863d
	.long	0x3ea7b0a2
	.long	0x3eb3c586
	.long	0x3cee1a31
	.long	0x3e563e5f
	.long	0x3eae7272
	.long	0xbe5bb00c
	.long	0xbdf9f43e
	.long	0xbb1c0250
	.long	0xbe67282b
	.long	0xbe250847
	.long	0x3db65ecc
	.long	0x3e51e535
	.long	0xbe2019bc
	.long	0xbe6711d2
	.long	0xbf0fe5ad
	.long	0xbd76b888
	.long	0x3ed2241a
	.long	0xbc6d5cf8
	.long	0xbef32958
	.long	0xbf1065f0
	.long	0xbdecabea
	.long	0x3e1181d7
	.long	0x3e8db0b6
	.long	0x3d2e7c3f
	.long	0xbc3d39b2
	.long	0x3c1edd0d
	.long	0x3e7c0552
	.long	0x3f202831
	.long	0x3f2eea62
	.long	0x3e71bc79
	.long	0xbe2a07eb
	.long	0xbeaae9bd
	.long	0xbe43c878
	.long	0x3d5b20dd
	.long	0x3e10f8fc
	.long	0x3e3b4290
	.long	0xbf0c4f10
	.long	0xbf3bb84d
	.long	0xbf1698b2
	.long	0xbec4b903
	.long	0x3f0a5009
	.long	0x3eb1378e
	.long	0xbe812c9d
	.long	0xbeb16d06
	.long	0xbe57fa26
	.long	0x3e616f29
	.long	0x3f28aeaf
	.long	0x3ebac4fe
	.long	0x3e047a8e
	.long	0xbde060be
	.long	0x3d76b085
	.long	0x3e9b0f8f
	.long	0x3e72ac78
	.long	0x3ef034b0
	.long	0x3e64a6ef
	.long	0x3dc8a547
	.long	0x3f0f3a64
	.long	0xbe5b9416
	.long	0xbeea54a9
	.long	0xbe58e8e4
	.long	0x3eba93ec
	.long	0x3ef96045
	.long	0xbed0bb56
	.long	0xbedbc069
	.long	0xbe7d7c01
	.long	0x3ed4718c
	.long	0x3e999b58
	.long	0xbeea7101
	.long	0xbe9b9ad0
	.long	0xbdb76059
	.long	0x3e8e5d34
	.long	0x3dea5add
	.long	0xbe307317
	.long	0xbeb40f25
	.long	0xbe1bad71
	.long	0x3eb284f8
	.long	0x3e3e4188
	.long	0x3d912d14
	.long	0x3e096593
	.long	0xbd3d2a3c
	.long	0xbcb4ed26
	.long	0xbe072c48
	.long	0x3dbbc5f8
	.long	0x3e6d2a56
	.long	0x3eadf2b0
	.long	0x3e2902c1
	.long	0x3ec0c271
	.long	0x3e47b5ee
	.long	0x3ea15dfa
	.long	0x3f113022
	.long	0xbdf1ac57
	.long	0x3e48af7a
	.long	0x3e07bc91
	.long	0x3d43bffd
	.long	0x3e08d9d1
	.long	0x3ce875f9
	.long	0xbe0e7d8e
	.long	0xbdf759fc
	.long	0xbe4a45e4
	.long	0xbf0095de
	.long	0xbd12bb71
	.long	0xbe42624b
	.long	0xbedae61a
	.long	0xbe91ca88
	.long	0xbe42eb19
	.size	.L__constant_8x1x5x5xf32, 800

	.section	".note.GNU-stack","",@progbits
