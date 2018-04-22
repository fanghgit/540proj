function read_cat_hier(filename, K)
    f = open(filename, "r")
    rows = zeros(Int, 0)
    cols = zeros(Int, 0)
    vals = zeros(Float64, 0)
    cc = 1
    for line in eachline(f)
        tmp = split(line, " ")
        if length(tmp) > 2
            error("more than 2 labels")
        end
        lbl1 = parse(Int, tmp[1])
        lbl2 = parse(Int, tmp[2])
        push!(rows, lbl1)
        push!(cols, lbl2)
        push!(vals, 1.0)
        push!(rows, lbl2)
        push!(cols, lbl1)
        push!(vals, 1.0)
    end
    adj_mat = sparse(rows, cols, vals, K, K)
    return adj_mat
end

function read(filename)
    f = open(filename, "r")
    #head = readline(f)
    #n, d, l = split(head, " ")
    #n = parse(Int, n)
    #d = parse(Int, d)
    #l = parse(Int, l)

    row_ft = zeros(Int, 0)
    col_ft = zeros(Int, 0)
    val_ft = zeros(Float64,0)

    row_lbl = zeros(Int, 0)
    col_lbl = zeros(Int, 0)
    val_lbl = zeros(Float64, 0)

    cc = 1
    for line in eachline(f)
        #println(cc)
        tmp = split(line, " ")
        if length(tmp) <= 1
            println("weird at line: ", cc)
            println(tmp)
	    continue
        end

        if tmp[1] != ""
            lbls = split(tmp[1], ",")
        else
            lbls = []
        end

        lbls = [ parse(Int, x) for x in lbls ]

        for lbl in lbls
            push!(row_lbl, lbl)
            push!(col_lbl, cc)
            push!(val_lbl, 1)
        end

	#println(tmp)
        for i = 2:length(tmp)
            if tmp[i] == ""
		continue
	    end
	    idx, val = split(tmp[i], ":")
            idx = parse(Int, idx)
            val = parse(Float64, val)
            push!(col_ft, idx )
            push!(val_ft, val)
            push!(row_ft, cc)
        end
        cc += 1
    end
    close(f)

    ft_mat = sparse( row_ft, col_ft, val_ft )
    lbl_mat = sparse( row_lbl, col_lbl, val_lbl )

    return ft_mat, lbl_mat


end
