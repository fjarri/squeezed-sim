<%def name="generate_input_state(
        kernel_declaration, alpha, beta, squeezing, decoherence, seed)">
<%
    real = dtypes.ctype(dtypes.real_for(alpha.dtype))
    comp = alpha.ctype
    s = ordering(representation)
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    const VSIZE_T sample_idx = virtual_global_id(0);
    const VSIZE_T mode_idx = virtual_global_id(1);
    const VSIZE_T flat_idx = virtual_global_flat_id();

    if (mode_idx > ${system.inputs}) {
        ${alpha.store_idx}(sample_idx, mode_idx, COMPLEX_CTR(${comp})(0, 0));
        ${beta.store_idx}(sample_idx, mode_idx, COMPLEX_CTR(${comp})(0, 0));
    }

    ${bijection.module}Key key = ${keygen.module}key_from_int(flat_idx);
    ${bijection.module}Counter ctr = ${bijection.module}make_counter_from_int(${seed});
    ${bijection.module}State st = ${bijection.module}make_state(key, ctr);

    ${real} sq = ${squeezing.load_idx}(mode_idx);
    ${real} eps = ${decoherence.load_idx}(mode_idx);

    ${real} sinh_sq = (${exp}(sq) - ${exp}(-sq)) / 2;
    ${real} cosh_sq = (${exp}(sq) + ${exp}(-sq)) / 2;
    ${real} n = sinh_sq * sinh_sq;
    ${real} m = (1 - eps) * cosh_sq * sinh_sq;

    ${sampler.module}Result w12 = ${sampler.module}sample(&st);
    ${real} w1 = w12.v[0];
    ${real} w2 = w12.v[1];

    ${comp} alpha, beta;

    ${real} x = sqrt((n + m + ${s}) / 2) * w1;
    ${real} y_scale = (n - m + ${s}) / 2; // this can be negative
    if (y_scale < 0) {
        ${real} y = sqrt(-y_scale) * w2;
        alpha = COMPLEX_CTR(${comp})(x - y, 0);
        beta = COMPLEX_CTR(${comp})(x + y, 0);
    }
    else {
        ${real} y = sqrt(y_scale) * w2;
        alpha = COMPLEX_CTR(${comp})(x, y);
        beta = COMPLEX_CTR(${comp})(x, -y);
    }

    ${alpha.store_idx}(sample_idx, mode_idx, alpha);
    ${beta.store_idx}(sample_idx, mode_idx, beta);
}
</%def>


<%def name="generate_apply_matrix_noise(kernel_declaration, w, seed)">
<%
    real = dtypes.ctype(dtypes.real_for(w.dtype))
    comp = w.ctype
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    const VSIZE_T sample_idx = virtual_global_id(0);
    const VSIZE_T mode_idx = virtual_global_id(1);
    const VSIZE_T flat_idx = virtual_global_flat_id();

    ${bijection.module}Key key = ${keygen.module}key_from_int(flat_idx);
    ${bijection.module}Counter ctr = ${bijection.module}make_counter_from_int(${seed});
    ${bijection.module}State st = ${bijection.module}make_state(key, ctr);

    ${sampler.module}Result w12 = ${sampler.module}sample(&st);
    ${real} w1 = w12.v[0];
    ${real} w2 = w12.v[1];

    ${w.store_idx}(sample_idx, mode_idx, COMPLEX_CTR(${comp})(w1, w2));
}
</%def>


<%def name="add_noise(kernel_declaration, alpha, beta, alpha_no_noise, noise)">
<%
    real = dtypes.ctype(dtypes.real_for(alpha.dtype))
    comp = alpha.ctype
%>

${kernel_declaration}
{
    VIRTUAL_SKIP_THREADS;

    const VSIZE_T sample_idx = virtual_global_id(0);
    const VSIZE_T mode_idx = virtual_global_id(1);

    ${comp} alpha_no_noise = ${alpha_no_noise.load_idx}(sample_idx, mode_idx);
    ${comp} noise = ${noise.load_idx}(sample_idx, mode_idx);
    ${comp} alpha = ${add}(alpha_no_noise, noise);
    ${comp} beta = ${conj}(alpha);

    ${alpha.store_idx}(sample_idx, mode_idx, alpha);
    ${beta.store_idx}(sample_idx, mode_idx, beta);
}
</%def>
