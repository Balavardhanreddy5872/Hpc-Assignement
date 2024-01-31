#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    const int size = 1000000;
    int* arr = (int*)malloc(size * sizeof(int));

    // Seed the random number generator
    srand(time(NULL));

    // Generate a random array
    for (int i = 0; i < size; i++) {
        arr[i] = rand();
    }

    clock_t start, end;
    start = clock();

    // Sort the array using Quick Sort
    quickSort(arr, 0, size - 1);

    end = clock();

    // Display the time taken for sorting
    // double time_taken = ((double)end - start) / CLOCKS_PER_SEC;
    // printf("Time taken for sorting: %f seconds\n", time_taken);

    // Don't forget to free the dynamically allocated memory
    free(arr);

    return 0;
}
