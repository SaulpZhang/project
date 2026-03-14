from z3 import *

# 定义变量
bucket_name = "compatible-special-format-0"
principal_id = "domain/233"
principal_type = "AssumedAgency"
action = "listBucket"

# 解析账户信息
buckets = [
    {
        "bucket_name": bucket_name,
        "bucket_policy": {
            "Statement": [
                {
                    "Action": [action],
                    "Condition": {
                        "stringequals": {
                            "g:PrincipalType": [principal_type]
                        }
                    },
                    "Effect": "Allow",
                    "Principal": {
                        "ID": [principal_id]
                    }
                }
            ]
        }
    }
]

# 构建约束
solver = Solver()

# 检查是否存在有效的授权配置
for bucket in buckets:
    policy = bucket["bucket_policy"]
    for statement in policy["Statement"]:
        action_match = statement["Action"] == [action]
        principal_type_match = statement["Condition"]["stringequals"]["g:PrincipalType"] == [principal_type]
        principal_id_match = statement["Principal"]["ID"] == [principal_id]
        effect_allow = statement["Effect"] == "Allow"
        solver.add(And(action_match, principal_type_match, principal_id_match, effect_allow))

# 求解
if solver.check() == sat:
    print("存在有效的授权配置")
else:
    print("不存在有效的授权配置")