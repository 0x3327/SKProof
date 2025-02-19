# import hashlib
# from hashlib import blake2s

# num = 3545663

# h = blake2s()
# h.update(num.to_bytes(3, 'big'))
# input_hash = h.hexdigest()

# print(input_hash)

# print(len("0xaf1ef82290a8721e2e681c76fd89b25763348c99e5983d155b98a15ddf3a95"))

# from poseidon.hash import Poseidon
# import poseidon

# poseidon_simple, t = poseidon.parameters.case_simple()

# input_vec = [x for x in range(0, t)]
# print("Input: ", input_vec)
# poseidon_digest = poseidon_simple.run_hash(input_vec)
# print("Output: ", hex(int(poseidon_digest)))

# input_vec = [0, 4937965, 94, 0, 3269761, 93, 0, 1260818, 94]
# for i in range(9):
#     print(i, input_vec[i].to_bytes(3, 'big'))

# security_level = 128
# input_rate = 5
# t = 12
# alpha = 5
# poseidon_new = poseidon.Poseidon(poseidon.parameters.prime_254, security_level, alpha, input_rate, t)

# input_vec_1 = [0, 1250167, 94, 0, 0, 100, 0, 0, 100]
# input_vec_2 = [0, 2608578, 94, 0, 0, 100, 0, 0, 100]
# print("Input 1: ", input_vec_1)
# print("Input 2: ", input_vec_2)
# poseidon_output_1 = poseidon_new.run_hash(input_vec_1)
# poseidon_output_2 = poseidon_new.run_hash(input_vec_2)
# print("output 1:", "0x{:064x}".format(int(poseidon_output_1)))
# print("output 2:", "0x{:064x}".format(int(poseidon_output_2)))

# o_1 = 16261083232639196931094147696935268331765025748922010601392071602570022994364
# o_2 = 4523589661846347754144087371588834264985015608349996127852847292915433788629

# print("output 1:", "0x{:064x}".format(o_1))
# print("output 2:", "0x{:064x}".format(o_2))

# from hashlib import blake2s

# input = ["0", "0", "100", "0", "0", "100", "0", "0", "100"]
# output = [0x0, 0x19fa69, 0x5d, 0x0, 0x228545, 0x5d, 0x0, 0x0, 0x64]

# h_in = blake2s()
# for i in input:
#     h_in.update(i.encode())
# input_hash = h_in.hexdigest()
# print(input_hash)

# h_out = blake2s()
# for o in output:
#     h_out.update(o.to_bytes(3, 'little'))
# output_hash = h_out.hexdigest()
# print(output_hash)

from poseidon_py.poseidon_hash import poseidon_hash_many

input = [0, 0, 100, 0, 0, 100, 0, 0, 100]
output = [0, 1702505, 93, 0, 2262341, 93, 0, 0, 100]

input_hash = poseidon_hash_many(input)
output_hash = poseidon_hash_many(output)

print(hex(int(input_hash)))
print(hex(int(output_hash)))

