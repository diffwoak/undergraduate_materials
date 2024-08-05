function  newtonMethod(f, f_prime, x0)
    epsilon = 1e-6; % 精度要求
    max_iterations = 1000; % 最大迭代次数
    iterations = 0;
    x = x0;
    
    while true
        fx = f(x);
        fx_prime = f_prime(x);
        
        if abs(fx_prime) < eps
            error("迭代失败：导数为零");
        end
        
        x_next = x - fx / fx_prime;
        iterations = iterations + 1;
        t=abs(x_next - x) ;
        disp("k= " + iterations+"  x= " + x+"  偏差："+ t);

        if t < epsilon || iterations >= max_iterations
            root = x_next;
            break;
        end
  
        x = x_next;
    end

    disp("近似根的值: " + root);
    disp("迭代次数: " + iterations);
end



