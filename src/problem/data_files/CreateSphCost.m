function Cost = CreateSphCost(model)
%CreateSphCost 此处显示有关此函数的摘要
%   此处显示详细说明
    Cost=@(x) SphCost(x,model);
end
