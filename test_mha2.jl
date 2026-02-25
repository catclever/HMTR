using Lux
m = MultiHeadAttention(16; nheads=2)
for m in methods(m)
    println(m)
end
