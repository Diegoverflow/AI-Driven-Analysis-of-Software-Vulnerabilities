import re
from gensim.models import FastText

# Define regular expressions for tokenizing
TOKEN_PATTERN = r'[@%]?\w+|[\[\]{}(),=*]|[<>]|[0-9]+|#\d+'

def tokenize_llvm_ir(llvm_code):
    # Apply regex to each line and extract tokens
    tokens = []
    for line in llvm_code.splitlines():
        line_tokens = re.findall(TOKEN_PATTERN, line)
        if line_tokens:
            tokens.append(line_tokens)
    return tokens


# Example LLVM code snippet (truncated for brevity)
llvm_code = """
define i32 @main(i32, i8**) #0 {
  call void @CWE121_Stack_Based_Buffer_Overflow__CWE805_wchar_t_declare_snprintf_18_good()
define void @CWE121_Stack_Based_Buffer_Overflow__CWE805_wchar_t_declare_snprintf_18_good() #0 {
  call void @goodG2B()
define internal void @goodG2B() #0 {
  %3 = alloca i32*, align 8
  %5 = alloca [100 x i32], align 16
  %6 = alloca [100 x i32], align 16
  %8 = getelementptr inbounds [100 x i32], [100 x i32]* %5, i32 0, i32 0
  store i32* %8, i32** %3, align 8
  %9 = load i32*, i32** %3, align 8
  %10 = getelementptr inbounds i32, i32* %9, i64 0
  store i32 0, i32* %10, align 4
  %11 = getelementptr inbounds [100 x i32], [100 x i32]* %6, i32 0, i32 0
  %13 = getelementptr inbounds [100 x i32], [100 x i32]* %6, i64 0, i64 99
  store i32 0, i32* %13, align 4
  %14 = load i32*, i32** %3, align 8
  %15 = bitcast i32* %14 to i8*
  %16 = getelementptr inbounds [100 x i32], [100 x i32]* %6, i32 0, i32 0
  %18 = load i32*, i32** %3, align 8
}
}
}
"""

# Tokenize the LLVM code
tokens = tokenize_llvm_ir(llvm_code)
for line_tokens in tokens:
    print(line_tokens)


# Load the FastText model
model = FastText.load('fasttext_model_isevc.bin')

# Example tokenized iSeVC (this is just one example; you'd likely have many)
tokenized_iSeVC = ['define', 'i32', '@main', '(', 'i32', ',', 'i8**', ')', '#0', '{']

# Get the vector representation for each token in the iSeVC
token_vectors = [model.wv[token] for token in tokenized_iSeVC if token in model.wv]

# Now token_vectors contains the 128-dimensional vector for each token in the sequence
print(token_vectors)  # This will be a list of 128-dimensional vectors