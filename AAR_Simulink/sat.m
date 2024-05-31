function S=sat(s) %±¥ºÍº¯Êý

    if abs(s)<=1
        S=s;
    else
        S=sign(s);
    end
end