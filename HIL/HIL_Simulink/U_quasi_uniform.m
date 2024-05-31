% 生成准均匀B样条曲线
function nodevector=U_quasi_uniform(n, k)
nodevector=zeros(1,n+k+1);% 节点数=控制点的个数+阶数
piecewise = n - k + 2;% B样条曲线的段数=控制点个数-次数
if piecewise==1% 只有一段曲线时，直接末尾重复度k
    nodevector(n+1+1:n+k+1)=1;
else
    for i=1:n-k+1% 中间段内节点均匀分布：两端共2k个节点，中间还剩(n+k+1-2k=n-k+1）个节点
        nodevector(k+i)=nodevector(k+i-1)+1/piecewise;
    end
    nodevector(n+1+1:n+k+1)=1;% 末尾重复度k
end
end