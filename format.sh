# WORKSPACE 格式化工具
buildifier WORKSPACE

find ./ -name 'BUILD' | xargs buildifier
