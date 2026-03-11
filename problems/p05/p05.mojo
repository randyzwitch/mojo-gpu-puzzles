from memory import UnsafePointer
from gpu import thread_idx
from gpu.host import DeviceContext
from testing import assert_equal

# ANCHOR: broadcast_add
comptime SIZE = 2
comptime BLOCKS_PER_GRID = 1
comptime THREADS_PER_BLOCK = (3, 3)
comptime dtype = DType.float32


fn broadcast_add(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    row = thread_idx.y
    col = thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if row < size and col < size:
        output[row * size + col] = a[col] + b[row]

#Output Positions
#Row[0], Col[0]: row * size + col = 0
#Row[0], Col[1]: row * size + col = 1
#Row[1], Col[0]: row * size + col = 2
#Row[1], Col[1]: row * size + col = 3

#Values
#a: [0,1]
#b: [0,10]

#RZ: The original way I expected a[row] + b[col] doesn't work...I assume this is a "row major" vs. "column major" difference?
#RZ: Easy here to guess by switching positions, but need to develop intuition why it need to be reversed

#From answer key
#[ a0 a1 ]  +  [ b0 ]  =  [ a0+b0  a1+b0 ]
#              [ b1 ]     [ a0+b1  a1+b1 ]


# ANCHOR_END: broadcast_add
def main() raises:
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](SIZE * SIZE)
        out.enqueue_fill(0)
        expected = ctx.enqueue_create_host_buffer[dtype](SIZE * SIZE)
        expected.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i + 1
                b_host[i] = i * 10

            for i in range(SIZE):
                for j in range(SIZE):
                    expected[i * SIZE + j] = a_host[j] + b_host[i]

        ctx.enqueue_function[broadcast_add, broadcast_add](
            out,
            a,
            b,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        ctx.synchronize()

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            for i in range(SIZE):
                for j in range(SIZE):
                    assert_equal(out_host[i * SIZE + j], expected[i * SIZE + j])
