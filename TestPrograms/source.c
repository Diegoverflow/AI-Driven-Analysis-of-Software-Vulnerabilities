#include <stdio.h>

// New unrelated function 1
void print_message() {
    printf("This is an unrelated message.\n");
}

// New unrelated function 2
int calculate_sum(int a, int b) {
    return a + b;
}

void modify_element(int *element) {
    *element = *element * 3;  // Example: tripling the element
}

void inner_function(int arr[], int size) {
    // Modify the array or perform operations on it
    for(int i = 0; i < size; i++) {
        arr[i] = arr[i] * 2;  // Example: doubling each element
        modify_element(&arr[i]);  // Further modify each element by calling modify_element
    }
}

int main() {
    // Call the unrelated functions
    print_message();  // Call to unrelated function 1

    int sum = calculate_sum(7, 5);  // Call to unrelated function 2
    printf("The sum is: %d\n", sum);

    int my_array[5] = {1, 2, 3, 4, 5};  // Define an array inside main
    int size = sizeof(my_array) / sizeof(my_array[0]);  // Calculate the size of the array

    inner_function(my_array, size);  // Pass the array and its size to the function

    // Print the modified array
    for(int i = 0; i < size; i++) {
        printf("%d ", my_array[i]);
    }
    printf("\n");

    return 0;
}
