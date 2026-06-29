import ctypes


lib = ctypes.CDLL("./libmath.so")


class SensorData(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("value", ctypes.c_double),
    ]


class Size(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
    ]


lib.square.argtypes = [ctypes.c_double]
lib.square.restype = ctypes.c_double

lib.scale_sensor.argtypes = [ctypes.POINTER(SensorData), ctypes.c_double]
lib.scale_sensor.restype = None

lib.sum_int_array.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.sum_int_array.restype = ctypes.c_int

lib.fill_message.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.fill_message.restype = None

Matrix2x3 = (ctypes.c_double * 3) * 2
lib.sum_matrix_2x3.argtypes = [Matrix2x3]
lib.sum_matrix_2x3.restype = ctypes.c_double

lib.fill_sequence.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
lib.fill_sequence.restype = None

lib.average_sensor_values.argtypes = [ctypes.POINTER(SensorData), ctypes.c_int]
lib.average_sensor_values.restype = ctypes.c_double

lib.string_length.argtypes = [ctypes.c_char_p]
lib.string_length.restype = ctypes.c_int

lib.make_size.argtypes = [ctypes.c_int, ctypes.c_int]
lib.make_size.restype = Size

lib.set_value_through_pointer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.set_value_through_pointer.restype = None

lib.redirect_pointer_to_value.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
    ctypes.POINTER(ctypes.c_int),
]
lib.redirect_pointer_to_value.restype = None

lib.create_sequence_heap.argtypes = [ctypes.c_int, ctypes.c_int]
lib.create_sequence_heap.restype = ctypes.POINTER(ctypes.c_int)

lib.free_memory.argtypes = [ctypes.c_void_p]
lib.free_memory.restype = None


print("square:", lib.square(5.0))

sensor = SensorData(id=101, value=12.5)
lib.scale_sensor(ctypes.byref(sensor), 1.6)
print("struct pointer:", sensor.id, sensor.value)

numbers = (ctypes.c_int * 5)(1, 2, 3, 4, 5)
print("static int array sum:", lib.sum_int_array(numbers, len(numbers)))

message = ctypes.create_string_buffer(32)
lib.fill_message(message, ctypes.sizeof(message))
print("char array:", message.value.decode("utf-8"))

matrix = Matrix2x3(
    (1.0, 2.0, 3.0),
    (4.0, 5.0, 6.0),
)
print("2D static array sum:", lib.sum_matrix_2x3(matrix))

length = 6
dynamic_array = (ctypes.c_int * length)()
lib.fill_sequence(dynamic_array, length, 10)
print("dynamic array from C:", list(dynamic_array))

sensors = (SensorData * 3)(
    SensorData(1, 10.0),
    SensorData(2, 15.5),
    SensorData(3, 20.0),
)
print("array of structs avg:", lib.average_sensor_values(sensors, len(sensors)))

text = b"ctypes input string"
print("char* input length:", lib.string_length(text))

size = lib.make_size(640, 480)
print("returned struct:", size.width, size.height)

value = ctypes.c_int(7)
lib.set_value_through_pointer(ctypes.byref(value), 99)
print("int* changed value:", value.value)

first_value = ctypes.c_int(111)
second_value = ctypes.c_int(222)
int_pointer = ctypes.pointer(first_value)
lib.redirect_pointer_to_value(ctypes.byref(int_pointer), ctypes.byref(second_value))
print("int** redirected pointer:", int_pointer.contents.value)

heap_length = 5
heap_pointer = lib.create_sequence_heap(heap_length, 50)
if heap_pointer:
    heap_values = [heap_pointer[i] for i in range(heap_length)]
    print("malloc/free array:", heap_values)
    lib.free_memory(heap_pointer)
