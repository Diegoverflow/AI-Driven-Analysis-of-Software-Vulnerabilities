from gensim.models import FastText

test = """
define i32 @main ( i32 , i8 ** ) { call void @CWE121_Buffer_Overflow_good ( ) }
define void @goodG2B ( ) { %3 = alloca i32 * , align 8 %5 = alloca [ 100 x i32 ] , align 16 }
"""


iSeVCs = [line.strip().split() for line in test]

print(iSeVCs)