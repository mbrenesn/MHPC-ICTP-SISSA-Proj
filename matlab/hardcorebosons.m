function loschmidt = hardcorebosons( ham_mat, times, v_initial, tolerance )

v_help = v_initial;
sizeits = size(times);
iterations = sizeits(2);

l_echo = dot(v_initial',v_initial);
loschmidt(1) = l_echo * conj(l_echo);
for tt=2:iterations
    w = expv(times(tt)-times(tt-1),ham_mat,v_help,tolerance,30);
    l_echo = dot(w',v_initial);
    loschmidt(tt) = l_echo * conj(l_echo);
    v_help = w;
end

end

