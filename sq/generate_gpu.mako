<%def name="generate_input_state(
        kernel_declaration, alpha, beta, squeezing, thermal_noise, input_transmission, seed)">
<%
    real = dtypes.ctype(dtypes.real_for(alpha.dtype))
    comp = alpha.ctype
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

    ${sampler.module}Result res;

    // Squeezing
    ${real} sq = (mode_idx < ${system.inputs}) ? ${squeezing.load_idx}(mode_idx) : 0;

    // Thermal
    ${real} thermal_re, thermal_im;
    if (mode_idx < ${system.inputs}) {
        ${real} n = ${thermal_noise.load_idx}(mode_idx);
        ${real} nts = sqrt(n / 2);

        ${sampler.module}Result w34 = ${sampler.module}sample(&st);
        ${real} w3 = w34.v[0];
        ${real} w4 = w34.v[1];
        thermal_re = nts * w3;
        thermal_im = nts * w4;
    }
    else {
        thermal_re = 0;
        thermal_im = 0;
    }

    ${comp} alpha, beta;

    %if representation == Representation.POSITIVE_P:

        if (mode_idx < ${system.inputs}) {
            ${sampler.module}Result w12 = ${sampler.module}sample(&st);
            ${real} w1 = w12.v[0];
            ${real} w2 = w12.v[1];

            ${real} sr = sinh(sq);
            ${real} cr = cosh(sq);
            ${real} a = sqrt(sr * (cr + 1) / 2);
            ${real} b = sqrt(sr * (cr - 1) / 2);

            alpha = COMPLEX_CTR(${comp})(a * w1 + b * w2 + thermal_re, thermal_im);
            beta = COMPLEX_CTR(${comp})(b * w1 + a * w2 + thermal_re, -thermal_im);
        }
        else {
            alpha = COMPLEX_CTR(${comp})(0, 0);
            beta = COMPLEX_CTR(${comp})(0, 0);
        }

    %elif representation == Representation.WIGNER:

        ${sampler.module}Result w12 = ${sampler.module}sample(&st);
        ${real} w1 = w12.v[0];
        ${real} w2 = w12.v[1];

        ${real} er = exp(sq);
        ${real} a = 0.5 * er;
        ${real} b = 0.5 / er;

        ${real} alpha_re = a * w1 + thermal_re;
        ${real} alpha_im = b * w2 + thermal_im;

        alpha = COMPLEX_CTR(${comp})(alpha_re, alpha_im);
        beta = COMPLEX_CTR(${comp})(alpha_re, -alpha_im);

    %elif representation == Representation.Q:

        ${sampler.module}Result w12 = ${sampler.module}sample(&st);
        ${real} w1 = w12.v[0];
        ${real} w2 = w12.v[1];

        ${real} e2r = exp(2 * sq);
        ${real} a = 0.5 * sqrt(e2r + 1);
        ${real} b = 0.5 * sqrt(1 / e2r + 1);

        ${real} alpha_re = a * w1 + thermal_re;
        ${real} alpha_im = b * w2 + thermal_im;

        alpha = COMPLEX_CTR(${comp})(alpha_re, alpha_im);
        beta = COMPLEX_CTR(${comp})(alpha_re, -alpha_im);

    %else:
    <%
        raise NotImplementedError(representation)
    %>
    %endif

    <%
        s = ordering(representation)
    %>

    %if not apply_transmission_coeffs:

        ${alpha.store_idx}(sample_idx, mode_idx, alpha);
        ${beta.store_idx}(sample_idx, mode_idx, beta);

    %else:

        ${real} itr = ${input_transmission.load_idx}(mode_idx);

        ${comp} alpha_i = ${mul_cr}(alpha, itr);
        ${comp} beta_i = ${mul_cr}(beta, itr);

        %if s != 0:
            ${sampler.module}Result w = ${sampler.module}sample(&st);

            ${real} coeff = sqrt((1 - itr * itr) * ${s / 2});
            ${real} w_re = w.v[0] * coeff;
            ${real} w_im = w.v[1] * coeff;

            alpha_i = ${add_cc}(alpha_i, COMPLEX_CTR(${comp})(w_re, w_im));
            beta_i = ${add_cc}(beta_i, COMPLEX_CTR(${comp})(w_re, -w_im));
        %endif

        ${alpha.store_idx}(sample_idx, mode_idx, alpha_i);
        ${beta.store_idx}(sample_idx, mode_idx, beta_i);

    %endif
}
</%def>


<%def name="output_transmission_transformation()">

</%def>
