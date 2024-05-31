offset5_origin4 = [400 0 -200];
offset5 = [400 0 -200];
t1 = 10;
t2 = 10;
v1 = 30;
v2 = 20;
R = v2 * t2 / 2;
for t = 1: 100
    t = mod(t, t1 + t2 + t1 + t2);
    if t <= t1
        offset5(1) = offset5_origin4(1) + v1 * t;
    elseif t <= t1 + t2
        offset5(2) = offset5_origin4(2) + v2 * (t - t1);
        offset5(1) = offset5_origin4(1) + v1 * t1 + R * sin(pi / 2 / R * offset5(2));
    elseif t <= t1 + t2 + t1
        offset5(1) = offset5_origin4(1) + v1 * t1 - v1 * (t - t1 - t2);
    elseif t <= t1 + t2 + t1 + t2
        offset5(2) = offset5_origin4(2) + v2 * t2 - v2 * (t - t1 - t2 - t1);
        offset5(1) = offset5_origin4(1) - R * sin(pi / 2 / R * offset5(2));
    end
    offset5
end
