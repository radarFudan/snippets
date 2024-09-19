import torch

for type in [torch.bfloat16, torch.float16, torch.float32, torch.float64]:
    type_precision = torch.tensor(torch.finfo(type).resolution, dtype=type)
    type_one = torch.tensor(1.0, dtype=type)
    type_below = torch.nextafter(type_one, torch.tensor(0.0, dtype=type))
    type_above = torch.nextafter(type_one, torch.tensor(2.0, dtype=type))

    type_ctx = torch.log(type_precision.to(torch.float64)) / torch.log(type_below.to(torch.float64))
    print(f"Use {type} to support context length {type_ctx:0.2e}")
    # print(f"Use {type} to support context length {type_ctx.to(torch.int64)}")