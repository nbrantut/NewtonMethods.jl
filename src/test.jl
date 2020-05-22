function myf(x)
    return [x[1]+x[2], 5*x[1]^2+3*x[2]-5]
end

function myj(x)
    return [1  1;
            10*x[1]  3]
end

x = newtonraphson(myf, [1,1], myj)
