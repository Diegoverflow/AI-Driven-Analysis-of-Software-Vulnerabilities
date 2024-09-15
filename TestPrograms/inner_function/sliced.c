// function declarations
void inner_function(unsigned int* var0, unsigned int var1);
int main(void);
void modify_element(unsigned int* var0);

// global variable definitions
unsigned char _str[31] = {84,104,105,115,32,105,115,32,97,110,32,117,110,114,101,108,97,116,101,100,32,109,101,115,115,97,103,101,46,10,0,};
unsigned char _str_1[16] = {84,104,101,32,115,117,109,32,105,115,58,32,37,100,10,0,};
unsigned int __const_main_my_array[5] = {1,2,3,4,5,};
unsigned char _str_2[4] = {37,100,32,0,};
unsigned char _str_3[2] = {10,0,};

void inner_function(unsigned int* var0, unsigned int var1){
    unsigned int* var2;
    unsigned int var3;
    unsigned int var4;
    block0:
    var2 = var0;
    var3 = var1;
    var4 = 0;
    goto block1;
    block1:
    if (((int)var4) < ((int)var3)) {
        (*(((unsigned int*)(var2)) + ((long)((int)var4)))) = (((int)(*(((unsigned int*)(var2)) + ((long)((int)var4))))) * ((int)2));
        modify_element(&(*(((unsigned int*)(var2)) + ((long)((int)var4)))));
        var4 = (((int)var4) + ((int)1));
        goto block1;
    } else {
        return;
    }
}

int main(void){
    unsigned int var0[5];
    unsigned int var1;
    block0:
    var0 = __const_main_my_array;
    var1 = 5;
    inner_function(&(var0[0]), var1);
    return 0;
}

void modify_element(unsigned int* var0){
    unsigned int* var1;
    block0:
    var1 = var0;
    (*var1) = (((int)(*var1)) * ((int)3));
    return;
}

