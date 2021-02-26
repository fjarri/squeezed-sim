<%def name="compound_click_probability_prepare(kernel_declaration, output, alpha, beta)">
<%
    real = dtypes.ctype(dtypes.real_for(alpha.dtype))
    comp = alpha.ctype
%>
${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;
    VSIZE_T sample_idx = virtual_global_id(0);
    VSIZE_T mode_idx = virtual_global_id(1);

    ${comp} alpha = ${alpha.load_idx}(sample_idx, mode_idx);
    ${comp} beta = ${beta.load_idx}(sample_idx, mode_idx);

    ${comp} t = ${mul_cc}(alpha, beta);
    ${comp} res = ${exp_c}(COMPLEX_CTR(${comp})(-t.x, -t.y));

    ${output.store_idx}(sample_idx, mode_idx, res);
}
</%def>

<%def name="compound_click_probability_aggregate(kernel_declaration, output, input)">
<%
    real = dtypes.ctype(dtypes.real_for(output.dtype))
    comp = output.ctype

    reads_per_thread = block_size // read_size
%>
${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    LOCAL_MEM ${comp} nps[${block_size}];

    VSIZE_T sample_idx = virtual_global_id(0);
    VSIZE_T output_idx = virtual_global_id(1);
    VSIZE_T thread_idx = virtual_local_id(1);

    ${comp} result = COMPLEX_CTR(${comp})(1, 0);
    ${real} theta = ${2 * numpy.pi / (max_total_clicks + 1)} * output_idx;
    ${comp} coeff = ${polar_unit}(-theta);

    for (int i = 0; i < ${full_steps}; i++) {
        VSIZE_T state_offset = i * ${block_size};

        %for read_idx in range(reads_per_thread):
        {
            VSIZE_T idx = ${read_idx * read_size} + thread_idx;
            nps[idx] = ${input.load_idx}(sample_idx, state_offset + idx);
        }
        %endfor

        LOCAL_BARRIER;

        if (output_idx < ${output_size}) {
            for (int i = 0; i < ${block_size}; i++) {
                ${comp} np = nps[i];
                ${comp} cp = COMPLEX_CTR(${comp})(1 - np.x, -np.y);
                ${comp} cpk = ${add_cc}(np, ${mul_cc}(cp, coeff));
                result = ${mul_cc}(result, cpk);
            }
        }

        LOCAL_BARRIER;
    }

    %if remainder_size != 0:
        VSIZE_T state_offset = ${full_steps * block_size};

        %for read_idx in range(reads_per_thread):
        {
            VSIZE_T idx = ${read_idx * read_size} + thread_idx;
            %if read_idx * read_size <= remainder_size:
                nps[idx] = ${input.load_idx}(sample_idx, state_offset + idx);
            %else:
                if (idx < ${remainder_size}) {
                    nps[idx] = ${input.load_idx}(sample_idx, state_offset + idx);
                }
            %endif
        }
        %endfor

        LOCAL_BARRIER;

        if (output_idx < ${output_size}) {
            for (int i = 0; i < ${remainder_size}; i++) {
                ${comp} np = nps[i];
                ${comp} cp = COMPLEX_CTR(${comp})(1 - np.x, -np.y);
                ${comp} cpk = ${add_cc}(np, ${mul_cc}(cp, coeff));
                result = ${mul_cc}(result, cpk);
            }
        }

        LOCAL_BARRIER;
    %endif

    if (output_idx < ${output_size}) {
        ${output.store_idx}(sample_idx, output_idx, result);
    }
}
</%def>
