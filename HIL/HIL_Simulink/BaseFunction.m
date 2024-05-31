
%定义第i个k阶B样条基函数
function Nik_u=BaseFunction(i, k, u, NodeVector)
if k==1% 定义Bi,0这一系列基函数
    if u>=NodeVector(i)&&u<NodeVector(i+1)
        Nik_u=1;
    else
        Nik_u=0;
    end
else
    % 公式中的两个分母
    denominator_1 = NodeVector(i + k - 1) - NodeVector(i);
    denominator_2 = NodeVector(i + k) - NodeVector( i + 1);
    % 如果遇到分母为0的情况，定义为1以便继续计算
    if denominator_1==0
        denominator_1=1;
    end
    if denominator_2==0
        denominator_2=1;
    end
    % 不断递归
    Nik_u=(u - NodeVector(i)) / denominator_1 * BaseFunction(i, k - 1, u, NodeVector) ...
    + (NodeVector(i + k) - u) / denominator_2 * BaseFunction(i + 1, k - 1, u, NodeVector);
end
end