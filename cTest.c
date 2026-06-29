void *malloc(unsigned long size);
void free(void *ptr);

typedef struct {
    int id;
    double value;
} SensorData;

typedef struct {
    int width;
    int height;
} Size;

double square(double x) {
    return x * x;
}

void scale_sensor(SensorData *sensor, double factor) {
    sensor->value *= factor;
}

int sum_int_array(const int *values, int length) {
    int total = 0;
    int i;

    for (i = 0; i < length; i++) {
        total += values[i];
    }

    return total;
}

void fill_message(char *buffer, int size) {
    const char *message = "Hello from C";
    int i = 0;

    if (size <= 0) {
        return;
    }

    while (i < size - 1 && message[i] != '\0') {
        buffer[i] = message[i];
        i++;
    }

    buffer[i] = '\0';
}

double sum_matrix_2x3(const double matrix[2][3]) {
    double total = 0.0;
    int row;
    int col;

    for (row = 0; row < 2; row++) {
        for (col = 0; col < 3; col++) {
            total += matrix[row][col];
        }
    }

    return total;
}

void fill_sequence(int *buffer, int length, int start) {
    int i;

    for (i = 0; i < length; i++) {
        buffer[i] = start + i;
    }
}

double average_sensor_values(const SensorData *sensors, int length) {
    double total = 0.0;
    int i;

    if (length <= 0) {
        return 0.0;
    }

    for (i = 0; i < length; i++) {
        total += sensors[i].value;
    }

    return total / length;
}

int string_length(const char *text) {
    int length = 0;

    while (text[length] != '\0') {
        length++;
    }

    return length;
}

Size make_size(int width, int height) {
    Size size;

    size.width = width;
    size.height = height;
    return size;
}

void set_value_through_pointer(int *value_ptr, int new_value) {
    if (value_ptr != 0) {
        *value_ptr = new_value;
    }
}

void redirect_pointer_to_value(int **target_ptr, int *new_location) {
    if (target_ptr != 0) {
        *target_ptr = new_location;
    }
}

int *create_sequence_heap(int length, int start) {
    int *buffer;
    int i;

    if (length <= 0) {
        return 0;
    }

    buffer = (int *)malloc((unsigned long)(sizeof(int) * length));
    if (buffer == 0) {
        return 0;
    }

    for (i = 0; i < length; i++) {
        buffer[i] = start + i;
    }

    return buffer;
}

void free_memory(void *ptr) {
    if (ptr != 0) {
        free(ptr);
    }
}

double sum_double_array(const double *values, int length) {
    double total = 0.0;
    int i;

    for (i = 0; i < length; i++) {
        total += values[i];
    }

    return total;
}

void scale_double_array(double *values, int length, double factor) {
    int i;

    for (i = 0; i < length; i++) {
        values[i] *= factor;
    }
}

double sum_matrix_flat(const double *matrix, int rows, int cols) {
    double total = 0.0;
    int i;

    for (i = 0; i < rows * cols; i++) {
        total += matrix[i];
    }

    return total;
}

void add_to_sensor_values(SensorData *sensors, int length, double delta) {
    int i;

    for (i = 0; i < length; i++) {
        sensors[i].value += delta;
    }
}
