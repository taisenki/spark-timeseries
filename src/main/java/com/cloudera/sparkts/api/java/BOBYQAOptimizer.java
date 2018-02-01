/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.cloudera.sparkts.api.java;

import java.util.Arrays;

import org.apache.commons.math.analysis.MultivariateRealFunction;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.MathIllegalStateException;
import org.apache.commons.math3.exception.MultiDimensionMismatchException;
import org.apache.commons.math3.exception.NoDataException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math.optimization.GoalType;
import org.apache.commons.math.optimization.MultivariateRealOptimizer;
import org.apache.commons.math.optimization.RealPointValuePair;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.optim.PointValuePair;
import org.apache.commons.math3.optim.nonlinear.scalar.MultivariateOptimizer;
import org.apache.commons.math3.util.FastMath;

/**
 * BOBYQA algorithm. This code is translated and adapted from the Fortran version
 * of this algorithm as implemented in http://plato.asu.edu/ftp/other_software/bobyqa.zip .
 * <em>http://</em>. <br>
 * See <em>http://www.optimization-online.org/DB_HTML/2010/05/2616.html</em>
 * for an introduction.
 *
 * <p>BOBYQA is particularly well suited for high dimensional problems
 * where derivatives are not available. In most cases it outperforms the
 * PowellOptimizer significantly. Stochastic algorithms like CMAESOptimizer
 * succeed more often than BOBYQA, but are more expensive. BOBYQA could
 * also be considered if you currently use a derivative based (Differentiable)
 * optimizer approximating the derivatives by finite differences.
 *
 * Comments of the subroutines were copied directly from the original sources.
 *
 * @version $Revision$ $Date$
 * @since 3.0
 */

public class BOBYQAOptimizer
        extends MultivariateOptimizer {

    /** Default value for {@link #rhobeg}: {@value} . */
    public static final double DEFAULT_INITIAL_RADIUS = 10.0;
    /** Default value for {@link #rhoend}: {@value} . */
    public static final double DEFAULT_STOPPING_RADIUS = 1E-8;

    public static final double HALF = .5;
    public static final double ONE = 1.;
    public static final double ONEMIN = -1.;
    public static final double TWO = 2.;
    public static final double ZERO = 0.;
    public static final double TEN = 10.;
    public static final double TENTH = .1;

    /** Differences between the upper and lower bounds. */
    private double[] boundDifference;

    /** goal (minimize or maximize) */
    private boolean isMinimize = true;

    /**
     * Number of objective variables/problem dimension.
     */
    private int n;  
    /**
     * number of interpolation conditions.
     */
    private final int npt;
    /**
     * npt + n.
     */
    private int ndim;
    /**
     * initial trust region radius..
     */
    private double rhobeg;
    /**
     * final trust region radius..
     */
    private double rhoend;    

    /** n + 1 */
    private int np;
    /** np - n */
    private int nptm;

    /**
     * argument vector.
     */
    private double[] x;
    /**
     * lower bound.
     */
    private double[] xl;
    /**
     * upper bound.
     */
    private double[] xu;

    // Working area
    
    /**
     * holds a shift of origin that should reduce the contributions
     *       from rounding errors to values of the model and Lagrange functions.
     */
    private double[] xbase;
    /**
     * is a two-dimensional array that holds the coordinates of the
     *       interpolation points relative to xbase.
     */
    private double[] xpt;
    /**
     * holds the values of F at the interpolation points.
     */
    private double[] fval;
    /**
     * is set to the displacement from xbase of the trust region centre.
     */
    private double[] xopt;
    /**
     * holds the gradient of the quadratic model at xbase+xopt
     */
    private double[] gopt;
    /**
     * holds the explicit second derivatives of the quadratic model.
     */
    private double[] hq;
    /**
     * contains the parameters of the implicit second derivatives of the
     *       quadratic model.
     */
    private double[] pq;
    /**
     * holds the last dimension columns of H
     */
    private double[] bmat;
    /**
     * holds the factorization of the leading npt by npt submatrix of H,
     *       this factorization being zmat times zmat^T, which provides both the
     *       correct rank and positive semi-definiteness.
     */
    private double[] zmat;
    /**
     * is the first dimension of bmat and has the value npt+n.
     */
    private double[] ndtm;
    /**
     * holds the differences xl-xbase.
     *       All the components of every xopt are going to satisfy the bounds
     *       sl[i] <=. xopt[i] <= su[i], with appropriate equalities when
     *       xopt is on a constraint boundary.
     */
    private double[] sl;
    /**
     * holds the differences xu-xbase.
     *       All the components of every xopt are going to satisfy the bounds
     *       sl[i] <=. xopt[i] <= su[i], with appropriate equalities when
     *       xopt is on a constraint boundary.
     */
    private double[] su; 
    /**
     * is chosen by trsbox or altmov. Usually xbase+xnew is the
     *       vector of variables for the next call of the evaluation function. xnew also satisfies
     *       the sl and su constraints in the way that has just been mentioned.
     */
    private double[] xnew;
    /**
     * is an alternative to xnew, chosen by altmov, that may replace xnew
     *       in order to increase the denominator in the updating of update.
     */
    private double[] xalt;
    /**
     * is reserved for a trial step from xopt, which is usually xnew-xopt.
     */
    private double[] d__;
    /**
     * contains the values of the Lagrange functions at a new point x.
     *       They are part of a product that requires vlag to be of length ndim
     */
    private double[] vlag;
    /**
     * holds the gradient of the quadratic model at xopt+d__. It is updated
     *       when d__ is updated
     */
    private double[] gnew;
    /**
     * is a working space vector. For i=0,1,...,N-1, the element xbdi[i] is
     *       set to -1.0, 0.0, or 1.0, the value being nonzero if and only if the
     *       i-th variable has become fixed at a bound, the bound being sl[i] or
     *       su[i] in the case xbdi[i]=-1.0 or xbdi[i]=1.0, respectively. This
     *       information is accumulated during the construction of xnew.
     */
    private double[] xbdi;
    /**
     * holds the current search direction.
     */
    private double[] s;
    /**
     * holds the changes in the gradient of q along w.
     */
    private double[] hs;
    /**
     * holds the reduced d__, where the reduced d__ is the same as d__,
     *       except that the components of the fixed variables are ZERO.
     */
    private double[] hred;
    /**
     * For J=1,2,...,N, PTSAUX(1,J) and
     *       PTSAUX(2,J) specify the two positions of provisional interpolation
     *       points when a nonzero step is taken along e_J (the J-th coordinate
     *       direction) through XBASE+XOPT, as specified below. Usually these
     *       steps have length DELTA, but other lengths are chosen if necessary
     *       in order to satisfy the given bounds on the variables.
     */
    private double[] ptsaux;
    /**
     * It has npt components that denote
     *       provisional new positions of the original interpolation points, in
     *       case changes are needed to restore the linear independence of the
     *       interpolation conditions. The K-th point is a candidate for change
     *       if and only if PTSID(K) is nonzero. In this case let p and q be the
     *       int parts of PTSID(K) and (PTSID(K)-p) multiplied by N+1. If p
     *       and q are both positive, the step from XBASE+XOPT to the new K-th
     *       interpolation point is PTSAUX(1,p)*e_p + PTSAUX(1,q)*e_q. Otherwise
     *       the step is PTSAUX(1,p)*e_p or PTSAUX(2,q)*e_q in the cases q=0 or
     *       p=0, respectively.
     */
    private double[] ptsid;
    /**
     * working space vector of length N for the gradient of the
     *       KNEW-th Lagrange function at XOPT.
     */
    private double[] glag;
    /**
     * working space vector of length NPT for the second derivative
     *       coefficients of the KNEW-th Lagrange function.
     */
    private double[] hcol;
    /**
     * working space vector of length 2N that is going to hold the
     *       constrained Cauchy step from XOPT of the Lagrange function, followed
     *       by the downhill version of XALT when the uphill step is calculated.
     */
    private double[] w;


    /**
     * @param numberOfInterpolationPoints Number of interpolation conditions.
     * For a problem of dimension {@code n}, its value must be in the interval
     * {@code [n+2, (n+1)(n+2)/2]}.
     * Choices that exceed {@code 2n+1} are not recommended.
     */
    public BOBYQAOptimizer(int numberOfInterpolationPoints) {
        this(numberOfInterpolationPoints,
                DEFAULT_INITIAL_RADIUS,
                DEFAULT_STOPPING_RADIUS);
    }

    /**
     * @param numberOfInterpolationPoints
     *            number of interpolation conditions. Its value must be for
     *            dimension=N in the interval [N+2,(N+1)(N+2)/2]. Choices that
     *            exceed 2*N+1 are not recommended. -1 means undefined, then
     *            2*N+1 is used as default.
     * @param initialTrustRegionRadius
     *            initial trust region radius.
     * @param stoppingTrustRegionRadius
     *            stopping trust region radius.
     */
    public BOBYQAOptimizer(int numberOfInterpolationPoints,
            double initialTrustRegionRadius,
            double stoppingTrustRegionRadius) {
        super(null); // No custom convergence criterion.
        this.npt = numberOfInterpolationPoints;
        this.rhobeg = initialTrustRegionRadius;
        this.rhoend = stoppingTrustRegionRadius;
    }

    /** {@inheritDoc} */
    @Override
    protected PointValuePair doOptimize() {
        final double[] lowerBound = getLowerBound();
        final double[] upperBound = getUpperBound();
        // -------------------- Initialization --------------------------------
        isMinimize = (getGoalType() == org.apache.commons.math3.optim.nonlinear.scalar.GoalType.MINIMIZE);
        final double[] guess = getStartPoint();
        // number of objective variables/problem dimension
        n = guess.length;
        checkParameters();
        x = guess.clone();
        if (lowerBound != null && upperBound != null) {
            xl = lowerBound;
            xu = upperBound;
            double minDiff = Double.POSITIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                boundDifference[i] = upperBound[i] - lowerBound[i];
                minDiff = FastMath.min(minDiff, boundDifference[i]);
            }
            if (minDiff < 2 * rhobeg)
                rhobeg = minDiff / 3.0;
        } else {
            xl = point(n, -1e300);
            xu = point(n, 1e300);
        }
        double value = bobyqa();
        return new PointValuePair(x,
                isMinimize ? value : -value);
    }

    /**
     *     This subroutine seeks the least value of a function of many variables,
     *     by applying a trust region method that forms quadratic models by
     *     interpolation. There is usually some freedom in the interpolation
     *     conditions, which is taken up by minimizing the Frobenius norm of
     *     the change to the second derivative of the model, beginning with the
     *     ZERO matrix. The values of the variables are constrained by upper and
     *     lower bounds. The arguments of the subroutine are as follows.
     *
     *     N must be set to the number of variables and must be at least two.
     *     NPT is the number of interpolation conditions. Its value must be in
     *       the interval [N+2,(N+1)(N+2)/2]. Choices that exceed 2*N+1 are not
     *       recommended.
     *     Initial values of the variables must be set in X(1),X(2),...,X(N). They
     *       will be changed to the values that give the least calculated F.
     *     For I=1,2,...,N, XL(I) and XU(I) must provide the lower and upper
     *       bounds, respectively, on X(I). The construction of quadratic models
     *       requires XL(I) to be strictly less than XU(I) for each I. Further,
     *       the contribution to a model from changes to the I-th variable is
     *       damaged severely by rounding errors if XU(I)-XL(I) is too small.
     *     RHOBEG and RHOEND must be set to the initial and final values of a trust
     *       region radius, so both must be positive with RHOEND no greater than
     *       RHOBEG. Typically, RHOBEG should be about one TENTH of the greatest
     *       expected change to a variable, while RHOEND should indicate the
     *       accuracy that is required in the final values of the variables. An
     *       error return occurs if any of the differences XU(I)-XL(I), I=1,...,N,
     *       is less than 2*RHOBEG.
     *     MAXFUN must be set to an upper bound on the number of calls of CALFUN.
     *     The array W will be used for working space. Its length must be at least
     *       (NPT+5)*(NPT+N)+3*N*(N+5)/2.
     *       
     * @return
     */
    private double bobyqa() {

        np = n + 1;
        nptm = npt - np;
        ndim = n + npt;
        int maxfun = getMaxEvaluations();

        // Partition the working space array, so that different parts of it can
        // be treated separately during the calculation of BOBYQB. The partition
        // requires the first (NPT+2)*(NPT+N)+3*N*(N+5)/2 elements of W plus the
        // space that is taken by the last array in the argument list of BOBYQB.
       
        xbase = new double[n];
        xpt = new double[n * npt];
        fval = new double[npt];
        xopt = new double[n];
        gopt = new double[n];
        hq = new double[n * np / 2];
        pq = new double[npt];
        bmat = new double[ndim * n];
        zmat = new double[npt * (npt - np)];
        sl = new double[n];
        su = new double[n]; 
        xnew = new double[n];
        xalt = new double[n];
        d__ = new double[n];
        vlag = new double[ndim];
        gnew = new double[np];
        xbdi = new double[n];
        s = new double[n];
        hs = new double[n];
        hred = new double[n];
        ptsaux = new double[2*n+1];
        ptsid = new double[npt];
        glag = new double[n];
        hcol = new double[npt];
        w = new double[2*ndim];
        
        // Return if there is insufficient space between the bounds. Modify the
        // initial X if necessary in order to avoid conflicts between the bounds
        // and the construction of the first quadratic model. The lower and upper
        // bounds on moves from the updated X are set now, in the ISL and ISU
        // partitions of W, in order to provide useful and exact information about
        // components of X that become within distance RHOBEG from their bounds.

        double ZERO = 0.;
        for (int j = 0; j < n; j++) {
            double temp = xu[j] - xl[j];
            sl[j] = xl[j] - x[j];
            su[j] = xu[j] - x[j];
            if (sl[j] >= -rhobeg) {
                if (sl[j] >= ZERO) {
                    x[j] = xl[j];
                    sl[j] = ZERO;
                    su[j] = temp;
                } else {
                    x[j] = xl[j] + rhobeg;
                    sl[j] = -rhobeg;
                    // Computing MAX
                    double d__1 = xu[j] - x[j];
                    su[j] = Math.max(d__1,rhobeg);
                }
            } else if (su[j] <= rhobeg) {
                if (su[j] <= ZERO) {
                    x[j] = xu[j];
                    sl[j] = -temp;
                    su[j] = ZERO;
                } else {
                    x[j] = xu[j] - rhobeg;
                    // Computing MIN
                    double d__1 = xl[j] - x[j];
                    double d__2 = -rhobeg;
                    sl[j] = Math.min(d__1,d__2);
                    su[j] = rhobeg;
                }
            }
        }

        // Make the call of BOBYQB.

        return bobyqb(maxfun);                     
    } // bobyqa

    // ----------------------------------------------------------------------------------------

    /**
     * @param maxfun
     * @return
     */
    private double bobyqb(int maxfun) {
        double nh = n * np / 2;
        int knew = 0;
        double adelt = 0;
        double dnorm = 0;
        double beta = 0;
        double distsq = 0;
        double f = 0;

        // The call of PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
        // BMAT and ZMAT for the first iteration, with the corresponding values of
        // of NF and KOPT, which are the number of calls of CALFUN so far and the
        // index of the interpolation point at the trust region centre. Then the
        // initial XOPT is set too. The branch to label 720 occurs if MAXFUN is
        // less than NPT. GOPT will be updated if KOPT is different from KBASE.

        IntRef nf = new IntRef(0);
        IntRef kopt = new IntRef(0);
        DoubleRef dsq = new DoubleRef(0);
        DoubleRef crvmin = new DoubleRef(0);
        DoubleRef cauchy = new DoubleRef(0);
        DoubleRef alpha = new DoubleRef(0);

        prelim(maxfun, nf, kopt);
        double xoptsq = ZERO;
        for (int i = 0; i < n; i++) {
            xopt[i] = xpt[kopt.value + i * npt];
            // Computing 2nd power
            double d__1 = xopt[i];
            xoptsq += d__1 * d__1;
        }
        double fsave = fval[0];
        if (nf.value < npt) { // should not happen
            throw new RuntimeException("Return from BOBYQA because the objective function has been called only " +
                    (nf.value+1) + " times.");
        }
        int kbase = 0;

        // Complete the settings that are required for the iterative procedure.

        double rho = rhobeg;
        double delta = rho;
        double nresc = nf.value;
        double ntrits = 0;
        double diffa = ZERO;
        double diffb = ZERO;
        int itest = 0;
        int nfsav = nf.value;
        double diffc = 0;
        double denom = 0;
        double ratio = 0;

        // Update GOPT if necessary before the first iteration and after each
        // call of RESCUE that makes a call of CALFUN.

        int state = 20;
        for(;;) L100 : switch (state) {
        case 20: {
            if (kopt.value != kbase) {
            	int ih = 0; 
                for (int j = 0; j < n; j++) {
                    for (int i = 0; i <= j; i++) {
                        if (i < j) {
                            gopt[j] = gopt[j] + hq[ih] * xopt[i];
                        }
                        gopt[i] = gopt[i] + hq[ih] * xopt[j];
                        ih++;
                    }
                }
                if (nf.value > npt) {
                    for (int k = 0; k < npt; k++) {
                        double temp = ZERO;
                         for (int j = 0; j < n; j++) {
                            temp += xpt[k + j * npt] * xopt[j];
                        }
                        temp = pq[k] * temp;
                        for (int i = 0; i < n; i++) {
                            gopt[i] += temp * xpt[k + i * npt];
                        }
                    }
                }
            }

            // Generate the next point in the trust region that provides a small value
            // of the quadratic model subject to the constraints on the variables.
            // The int NTRITS is set to the number "trust region" iterations that
            // have occurred since the last "alternative" iteration. If the length
            // of XNEW-XOPT is less than HALF*RHO, however, then there is a branch to
            // label 650 or 680 with NTRITS=-1, instead of calculating F at XNEW.

        }
        case 60: {
            trsbox(delta, dsq, crvmin);

            // Computing MIN
            double d__1 = delta;
            double d__2 = Math.sqrt(dsq.value);
            dnorm = Math.min(d__1,d__2);
            if (dnorm < HALF * rho) {
                ntrits = -1;
                // Computing 2nd power
                d__1 = TEN * rho;
                distsq = d__1 * d__1;
                if (nf.value <= nfsav + 2) {
                    state = 650; break;
                }

                // The following choice between labels 650 and 680 depends on whether or
                // not our work with the current RHO seems to be complete. Either RHO is
                // decreased or termination occurs if the errors in the quadratic model at
                // the last three interpolation points compare favourably with predictions
                // of likely improvements to the model within distance HALF*RHO of XOPT.

                // Computing MAX
                d__1 = Math.max(diffa,diffb);
                double errbig = Math.max(d__1,diffc);
                double frhosq = rho * .125 * rho;
                if (crvmin.value > ZERO && errbig > frhosq * crvmin.value) {
                    state = 650; break;
                }
                double bdtol = errbig / rho;
                for (int j = 0; j < n; j++) {
                    double bdtest = bdtol;
                    if (xnew[j] == sl[j]) {
                        bdtest = w[j];
                    }
                    if (xnew[j] == su[j]) {
                        bdtest = -w[j];
                    }
                    if (bdtest < bdtol) {
                    	int hj = j+1;
                        double curv = hq[(hj + hj * hj) / 2 - 1];
                        for (int k = 0; k < npt; k++) {
                            // Computing 2nd power
                            d__1 = xpt[k + j * npt];
                            curv += pq[k] * (d__1 * d__1);
                        }
                        bdtest += HALF * curv * rho;
                        if (bdtest < bdtol) {
                            state = 650; break L100;
                        }
                    }
                }
                state = 680; break;
            }
            ++ntrits;

            // Severe cancellation is likely to occur if XOPT is too far from XBASE.
            // If the following test holds, then XBASE is shifted so that XOPT becomes
            // ZERO. The appropriate changes are made to BMAT and to the second
            // derivatives of the current model, beginning with the changes to BMAT
            // that do not depend on ZMAT. VLAG is used temporarily for working space.

        }
        case 90: {
            if (dsq.value <= xoptsq * .001) {
                double fracsq = xoptsq * .25;
                double sumpq = ZERO;
                for (int k = 0; k < npt; k++) {
                    sumpq += pq[k];
                    double sum = -HALF * xoptsq;
                    for (int i = 0; i < n; i++) {
                        sum += xpt[k + i * npt] * xopt[i];
                    }
                    w[npt+k] = sum;
                    double temp = fracsq - HALF * sum;
                    for (int i = 0; i < n; i++) {
                        w[i] = bmat[k + i * ndim];
                        vlag[i] = sum * xpt[k + i * npt] + temp * xopt[i];
                        int ip = npt + i;
                        for (int j = 0; j <= i; j++) {
                            bmat[ip + j * ndim] = bmat[ip + j * ndim] + w[i] * vlag[j] + vlag[i] * w[j];
                            //bmat[ip + j * ndim] += w[i] * vlag[j] + vlag[i] * w[j]; //???
                        }
                    }
                }

                // Then the revisions of BMAT that depend on ZMAT are calculated.

                for (int jj = 0; jj < nptm; jj++) {
                    double sumz = ZERO;
                    double sumw = ZERO;
                    for (int k = 0; k < npt; k++) {
                        sumz += zmat[k + jj * npt];
                        vlag[k] = w[npt+k] * zmat[k + jj * npt];
                        sumw += vlag[k];
                    }
                    for (int j = 0; j < n; j++) {
                        double sum = (fracsq * sumz - HALF * sumw) * xopt[j];
                        for (int k = 0; k < npt; k++) {
                            sum += vlag[k] * xpt[k + j * npt];
                        }
                        w[j] = sum;
                        for (int k = 0; k < npt; k++) {
                            bmat[k + j * ndim] +=
                                    sum * zmat[k + jj * npt];
                        }
                    }
                    for (int i = 0; i < n; i++) {
                        int ip = i + npt;
                        double temp = w[i];
                        for (int j = 0; j <= i; j++) {
                            bmat[ip + j * ndim] += temp * w[j];
                        }
                    }
                }

                // The following instructions complete the shift, including the changes
                // to the second derivative parameters of the quadratic model.

                int ih = 0;
                for (int j = 0; j < n; j++) {
                    w[j] = -HALF * sumpq * xopt[j];
                    for (int k = 0; k < npt; k++) {
                        w[j] = w[j] + pq[k] * xpt[k + j * npt];
                        xpt[k + j * npt] -= xopt[j];
                    }
                    for (int i = 0; i <= j; i++) {
                        hq[ih] = hq[ih] + w[i] * xopt[j] + xopt[i] * w[j];
                        //hq[ih] += w[i] * xopt[j] + xopt[i] * w[j]; //???
                        bmat[npt + i + j * ndim] = bmat[npt + j + i * ndim];
                        ih++;
                    }
                }
                for (int i = 0; i < n; i++) {
                    xbase[i] = xbase[i] + xopt[i];
                    xnew[i] = xnew[i] - xopt[i];
                    sl[i] = sl[i] - xopt[i];
                    su[i] = su[i] - xopt[i];
                    xopt[i] = ZERO;
                }
                xoptsq = ZERO;
            }
            if (ntrits == 0) {
                state = 210; break;
            }
            state = 230; break;

            // XBASE is also moved to XOPT by a call of RESCUE. This calculation is
            // more expensive than the previous shift, because new matrices BMAT and
            // ZMAT are generated from scratch, which may include the replacement of
            // interpolation points whose positions seem to be causing near linear
            // dependence in the interpolation conditions. Therefore RESCUE is called
            // only if rounding errors have reduced by at least a factor of two the
            // denominator of the formula for updating the H matrix. It provides a
            // useful safeguard, but is not invoked in most applications of BOBYQA.

        }
        case 190: {
            nfsav = nf.value;
            kbase = kopt.value;

            rescue(nf, delta, kopt);

            // XOPT is updated now in case the branch below to label 720 is taken.
            // Any updating of GOPT occurs after the branch below to label 20, which
            // leads to a trust region iteration as does the branch to label 60.

            xoptsq = ZERO;
            if (kopt.value != kbase) {
                for (int i = 0; i < n; i++) {
                    xopt[i] = xpt[kopt.value + i * npt];
                    // Computing 2nd power
                    double d__1 = xopt[i];
                    xoptsq += d__1 * d__1;
                }
            }
            nresc = nf.value;
            if (nfsav < nf.value) {
                nfsav = nf.value;
                state = 20; break;
            }
            if (ntrits > 0) {
                state = 60; break;
            }
        }
        case 210: {
            // Pick two alternative vectors of variables, relative to XBASE, that
            // are suitable as new positions of the KNEW-th interpolation point.
            // Firstly, XNEW is set to the point on a line through XOPT and another
            // interpolation point that minimizes the predicted value of the next
            // denominator, subject to ||XNEW - XOPT|| .LEQ. ADELT and to the SL
            // and SU bounds. Secondly, XALT is set to the best feasible point on
            // a constrained version of the Cauchy step of the KNEW-th Lagrange
            // function, the corresponding value of the square of this function
            // being returned in CAUCHY. The choice between these alternatives is
            // going to be made when the denominator is calculated.

            altmov(kopt.value, knew, adelt, alpha, cauchy);
            for (int i = 0; i < n; i++) {
                d__[i] = xnew[i] - xopt[i];
            }

            // Calculate VLAG and BETA for the current choice of D. The scalar
            // product of D with XPT(K,.) is going to be held in W(NPT+K) for
            // use when VQUAD is calculated.

        }
        case 230: {
            for (int k = 0; k < npt; k++) {
                double suma = ZERO;
                double sumb = ZERO;
                double sum = ZERO;
                for (int j = 0; j < n; j++) {
                    suma += xpt[k + j * npt] * d__[j];
                    sumb += xpt[k + j * npt] * xopt[j];
                    sum += bmat[k + j * ndim] * d__[j];
                }
                w[k] =  suma * (HALF * suma + sumb);
                vlag[k] = sum;
                w[npt + k] = suma;
            }
            beta = ZERO;
            for (int jj = 0; jj < nptm; jj++) {
                double sum = ZERO;
                for (int k = 0; k < npt; k++) {
                    sum += zmat[k + jj * npt] * w[k];
                }
                beta -= sum * sum;
                for (int k = 0; k < npt; k++) {
                    vlag[k] += sum * zmat[k + jj * npt];
                }
            }
            dsq.value = ZERO;
            double bsum = ZERO;
            double dx = ZERO;
            for (int j = 0; j < n; j++) {
                // Computing 2nd power
                double d__1 = d__[j];
                dsq.value += d__1 * d__1;
                double sum = ZERO;
                for (int k = 0; k < npt; k++) {
                    sum += w[k] * bmat[k + j * ndim];
                }
                bsum += sum * d__[j];
                int jp = npt + j;
                for (int i = 0; i < n; i++) {
                    sum += bmat[jp + i * ndim] * d__[i];
                }
                vlag[jp] = sum;
                bsum += sum * d__[j];
                dx += d__[j] * xopt[j];
            }
            beta = dx * dx + dsq.value * (xoptsq + dx + dx + HALF * dsq.value) + beta - bsum;
            vlag[kopt.value] += ONE;
            
            // If NTRITS is ZERO, the denominator may be increased by replacing
            // the step D of ALTMOV by a Cauchy step. Then RESCUE may be called if
            // rounding errors have damaged the chosen denominator.

            if (ntrits == 0) {
                // Computing 2nd power
                double d__1 = vlag[knew];
                denom = d__1 * d__1 + alpha.value * beta;
                if (denom < cauchy.value && cauchy.value > ZERO) {
                    for (int i = 0; i < n; i++) {
                        xnew[i] = xalt[i];
                        d__[i] = xnew[i] - xopt[i];
                    }
                    cauchy.value = ZERO;
                    state = 230; break;
                }
                // Computing 2nd power
                d__1 = vlag[knew];
                if (denom <= HALF * (d__1 * d__1)) {
                    if (nf.value > nresc) {
                        state = 190; break;
                    }
                    throw new MathIllegalStateException(LocalizedFormats.TOO_SMALL_INTEGRATION_INTERVAL, nf.value);
                }

                // Alternatively, if NTRITS is positive, then set KNEW to the index of
                // the next interpolation point to be deleted to make room for a trust
                // region step. Again RESCUE may be called if rounding errors have damaged
                // the chosen denominator, which is the reason for attempting to select
                // KNEW before calculating the next value of the objective function.

            } else {
                double delsq = delta * delta;
                double scaden = ZERO;
                double biglsq = ZERO;
                knew = 0;
                for (int k = 0; k < npt; k++) {
                    if (k == kopt.value) {
                        continue;
                    }
                    double hdiag = ZERO;
                    for (int jj = 0; jj < nptm; jj++) {
                        // Computing 2nd power
                        double d__1 = zmat[k + jj * npt];
                        hdiag += d__1 * d__1;
                    }
                    // Computing 2nd power
                    double d__1 = vlag[k];
                    double den = beta * hdiag + d__1 * d__1;
                    distsq = ZERO;
                    for (int j = 0; j < n; j++) {
                        // Computing 2nd power
                        d__1 = xpt[k + j * npt] - xopt[j];
                        distsq += d__1 * d__1;
                    }
                    // Computing MAX
                    // Computing 2nd power
                    double d__3 = distsq / delsq;
                    d__1 = ONE;
                    double d__2 = d__3 * d__3;
                    double temp = Math.max(d__1,d__2);
                    if (temp * den > scaden) {
                        scaden = temp * den;
                        knew = k;
                        denom = den;
                    }
                    // Computing MAX
                    // Computing 2nd power
                    d__3 = vlag[k];
                    d__1 = biglsq;
                    d__2 = temp * (d__3 * d__3);
                    biglsq = Math.max(d__1,d__2);
                }
                if (scaden <= HALF * biglsq) {
                    if (nf.value > nresc) {
                        state = 190; break;
                    }
                    throw new MathIllegalStateException(LocalizedFormats.TOO_SMALL_INTEGRATION_INTERVAL, nf.value);
                }
            }

            // Put the variables for the next calculation of the objective function
            //   in XNEW, with any adjustments for the bounds.

            // Calculate the value of the objective function at XBASE+XNEW, unless
            //   the limit on the number of calculations of F has been reached.

        }
        case 360: {
            for (int i = 0; i < n; i++) {
                // Computing MIN
                // Computing MAX
                double d__3 = xl[i];
                double d__4 = xbase[i] + xnew[i];
                double d__1 = Math.max(d__3,d__4);
                double d__2 = xu[i];
                x[i] = Math.min(d__1,d__2);
                if (xnew[i] == sl[i]) {
                    x[i] = xl[i];
                }
                if (xnew[i] == su[i]) {
                    x[i] = xu[i];
                }
            }
            if (nf.value > maxfun) { // should not happen,
                // TooManyEvaluationsException is thrown before
                throw new RuntimeException("Return from BOBYQA because the objective function has been called max_f_evals times.");
            }
            nf.value++;
            f = computeObjectiveValue(x);
            if (!isMinimize)
                f = -f;
            
//        	if (nf.value >= 24)
//        		nf.value = nf.value;
            
            if (ntrits == -1) {
                fsave = f;
                state = 720; break;
            }
            
            // Use the quadratic model to predict the change in F due to the step D,
            //   and set DIFF to the error of this prediction.

            double fopt = fval[kopt.value];
            double vquad = ZERO;
            int ih = 0;
            for (int j = 0; j < n; j++) {
                vquad += d__[j] * gopt[j];
                for (int i = 0; i <= j; i++) {
                    double temp = d__[i] * d__[j];
                    if (i == j) {
                        temp = HALF * temp;
                    }
                    vquad += hq[ih] * temp;
                    ih++;
                }
            }
            for (int k = 0; k < npt; k++) {
                // Computing 2nd power
                double d__1 = w[npt + k];
                vquad += HALF * pq[k] * (d__1 * d__1);
            }
            double diff = f - fopt - vquad;
            diffc = diffb;
            diffb = diffa;
            diffa = Math.abs(diff);
            if (dnorm > rho) {
                nfsav = nf.value;
            }

            // Pick the next value of DELTA after a trust region step.

            if (ntrits > 0) {
                if (vquad >= ZERO) {
                    throw new MathIllegalStateException(LocalizedFormats.TRUST_REGION_STEP_FAILED, vquad);
                }
                ratio = (f - fopt) / vquad;
                if (ratio <= TENTH) {
                    // Computing MIN
                    double d__1 = HALF * delta;
                    delta = Math.min(d__1,dnorm);
                } else if (ratio <= .7) {
                    // Computing MAX
                    double d__1 = HALF * delta;
                    delta = Math.max(d__1,dnorm);
                } else {
                    // Computing MAX
                    double d__1 = HALF * delta;
                    double d__2 = dnorm + dnorm;
                    delta = Math.max(d__1,d__2);
                }
                if (delta <= rho * 1.5) {
                    delta = rho;
                }

                // Recalculate KNEW and DENOM if the new F is less than FOPT.

                if (f < fopt) {
                    int ksav = knew;
                    double densav = denom;
                    double delsq = delta * delta;
                    double scaden = ZERO;
                    double biglsq = ZERO;
                    knew = 0;
                    for (int k = 0; k < npt; k++) {
                        double hdiag = ZERO;
                        for (int jj = 0; jj < nptm; jj++) {
                            // Computing 2nd power
                            double d__1 = zmat[k + jj * npt];
                            hdiag += d__1 * d__1;
                        }
                        // Computing 2nd power
                        double d__1 = vlag[k];
                        double den = beta * hdiag + d__1 * d__1;
                        distsq = ZERO;
                         for (int j = 0; j < n; j++) {
                            // Computing 2nd power
                            d__1 = xpt[k + j * npt] - xnew[j];
                            distsq += d__1 * d__1;
                        }
                        // Computing MAX
                        // Computing 2nd power
                        double d__3 = distsq / delsq;
                        d__1 = ONE;
                        double d__2 = d__3 * d__3;
                        double temp = Math.max(d__1,d__2);
                        if (temp * den > scaden) {
                            scaden = temp * den;
                            knew = k;
                            denom = den;
                        }
                        // Computing MAX
                        // Computing 2nd power
                        d__3 = vlag[k];
                        d__1 = biglsq;
                        d__2 = temp * (d__3 * d__3);
                        biglsq = Math.max(d__1,d__2);
                    }
                    if (scaden <= HALF * biglsq) {
                        knew = ksav;
                        denom = densav;
                    }
                }
            }

            // Update BMAT and ZMAT, so that the KNEW-th interpolation point can be
            // moved. Also update the second derivative terms of the model.

            update(beta, denom, knew);

            ih = 0;
            double pqold = pq[knew];
            pq[knew] = ZERO;
            for (int i = 0; i < n; i++) {
                double temp = pqold * xpt[knew + i * npt];
                for (int j = 0; j <= i; j++) {
                    hq[ih] += temp * xpt[knew + j * npt];
                    ih++;
                }
            }
            for (int jj = 0; jj < nptm; jj++) {
                double temp = diff * zmat[knew + jj * npt];
                for (int k = 0; k < npt; k++) {
                    pq[k] += temp * zmat[k + jj * npt];
                }
            }

            // Include the new interpolation point, and make the changes to GOPT at
            // the old XOPT that are caused by the updating of the quadratic model.

            fval[knew] = f;
            for (int i = 0; i < n; i++) {
                xpt[knew + i * npt] = xnew[i];
                w[i] = bmat[knew + i * ndim];
            }
            for (int k = 0; k < npt; k++) {
                double suma = ZERO;
                for (int jj = 0; jj < nptm; jj++) {
                    suma += zmat[knew + jj * npt] * zmat[k + jj * npt];
                }
                double sumb = ZERO;
                for (int j = 0; j < n; j++) {
                    sumb += xpt[k + j * npt] * xopt[j];
                }
                double temp = suma * sumb;
                for (int i = 0; i < n; i++) {
                    w[i] += temp * xpt[k + i * npt];
                }
            }
            for (int i = 0; i < n; i++) {
                gopt[i] = gopt[i] + diff * w[i];
            }

            // Update XOPT, GOPT and KOPT if the new calculated F is less than FOPT.

            if (f < fopt) {
                kopt.value = knew;
                xoptsq = ZERO;
                ih = 0;
                for (int j = 0; j < n; j++) {
                    xopt[j] = xnew[j];
                    // Computing 2nd power
                    double d__1 = xopt[j];
                    xoptsq += d__1 * d__1;
                    for (int i = 0; i <= j; i++) {
                        if (i < j) {
                            gopt[j] += hq[ih] * d__[i];
                        }
                        gopt[i] += hq[ih] * d__[j];
                        ih++;
                    }
                }
                for (int k = 0; k < npt; k++) {
                    double temp = ZERO;
                    for (int j = 0; j < n; j++) {
                        temp += xpt[k + j * npt] * d__[j];
                    }
                    temp = pq[k] * temp;
                    for (int i = 0; i < n; i++) {
                        gopt[i] = gopt[i] + temp * xpt[k + i * npt];
                    }
                }
            }

            // Calculate the parameters of the least Frobenius norm interpolant to
            // the current data, the gradient of this interpolant at XOPT being put
            // into VLAG(NPT+I), I=1,2,...,N.

            if (ntrits > 0) {
                 for (int k = 0; k < npt; k++) {
                    vlag[k] = fval[k] - fval[kopt.value];
                    w[k] = ZERO;
                }
                for (int j = 0; j < nptm; j++) {
                    double sum = ZERO;
                     for (int k = 0; k < npt; k++) {
                        sum += zmat[k + j * npt] * vlag[k];
                    }
                    for (int k = 0; k < npt; k++) {
                        w[k] += sum * zmat[k + j * npt];
                    }
                }
                for (int k = 0; k < npt; k++) {
                    double sum = ZERO;
                     for (int j = 0; j < n; j++) {
                        sum += xpt[k + j * npt] * xopt[j];
                    }
                    w[k + npt] = w[k];
                    w[k] = sum * w[k];
                }
                double gqsq = ZERO;
                double gisq = ZERO;
                for (int i = 0; i < n; i++) {
                    double sum = ZERO;
                    for (int k = 0; k < npt; k++) {
                        sum = sum + bmat[k + i * ndim] *
                                  vlag[k] + xpt[k + i * npt] * w[k];
                        //sum += bmat[k + i * ndim] *
                        //		  vlag[k] + xpt[k + i * npt] * w[k]; //???
                    }
                    if (xopt[i] == sl[i]) {
                        // Computing MIN
                        double d__2 = ZERO;
                        double d__3 = gopt[i];
                        // Computing 2nd power
                        double d__1 = Math.min(d__2,d__3);
                        gqsq += d__1 * d__1;
                        // Computing 2nd power
                        d__1 = Math.min(ZERO,sum);
                        gisq += d__1 * d__1;
                    } else if (xopt[i] == su[i]) {
                        // Computing MAX
                        double d__2 = ZERO;
                        double d__3 = gopt[i];
                        // Computing 2nd power
                        double d__1 = Math.max(d__2,d__3);
                        gqsq += d__1 * d__1;
                        // Computing 2nd power
                        d__1 = Math.max(ZERO,sum);
                        gisq += d__1 * d__1;
                    } else {
                        // Computing 2nd power
                        double d__1 = gopt[i];
                        gqsq += d__1 * d__1;
                        gisq += sum * sum;
                    }
                    vlag[npt + i] = sum;
                }

                // Test whether to replace the new quadratic model by the least Frobenius
                // norm interpolant, making the replacement if the test is satisfied.

                ++itest;
                if (gqsq < TEN * gisq) {
                    itest = 0;
                }
                if (itest >= 3) {
                    double i1 = Math.max(npt,nh);
                    for (int i = 0; i < i1; i++) {
                        if (i < n) {
                            gopt[i] = vlag[npt + i];
                        }
                        if (i < npt) {
                            pq[i] = w[npt + i];
                        }
                        if (i < nh) {
                            hq[i] = ZERO;
                        }
                        itest = 0;
                    }
                }
            }

            // If a trust region step has provided a sufficient decrease in F, then
            // branch for another trust region calculation. The case NTRITS=0 occurs
            // when the new interpolation point was reached by an alternative step.

            if (ntrits == 0) {
                state = 60; break;
            }
            if (f <= fopt + TENTH * vquad) {
                state = 60; break;
            }

            // Alternatively, find out if the interpolation points are close enough
            //   to the best point so far.

            // Computing MAX
            // Computing 2nd power
            double d__3 = TWO * delta;
            // Computing 2nd power
            double d__4 = TEN * rho;
            double d__1 = d__3 * d__3;
            double d__2 = d__4 * d__4;
            distsq = Math.max(d__1,d__2);
        }
        case 650: {
            knew = -1;
            for (int k = 0; k < npt; k++) {
                double sum = ZERO;
                for (int j = 0; j < n; j++) {
                    // Computing 2nd power
                    double d__1 = xpt[k + j * npt] - xopt[j];
                    sum += d__1 * d__1;
                }
                if (sum > distsq) {
                    knew = k;
                    distsq = sum;
                }
            }

            // If KNEW is positive, then ALTMOV finds alternative new positions for
            // the KNEW-th interpolation point within distance ADELT of XOPT. It is
            // reached via label 90. Otherwise, there is a branch to label 60 for
            // another trust region iteration, unless the calculations with the
            // current RHO are complete.

            if (knew >= 0) {
                double dist = Math.sqrt(distsq);
                if (ntrits == -1) {
                    // Computing MIN
                    double d__1 = TENTH * delta;
                    double d__2 = HALF * dist;
                    delta = Math.min(d__1,d__2);
                    if (delta <= rho * 1.5) {
                        delta = rho;
                    }
                }
                ntrits = 0;
                // Computing MAX
                // Computing MIN
                double d__2 = TENTH * dist;
                double d__1 = Math.min(d__2,delta);
                adelt = Math.max(d__1,rho);
                dsq.value = adelt * adelt;
                state = 90; break;
            }
            if (ntrits == -1) {
                state = 680; break;
            }
            if (ratio > ZERO) {
                state = 60; break;
            }
            if (Math.max(delta,dnorm) > rho) {
                state = 60; break;
            }

            // The calculations with the current value of RHO are complete. Pick the
            //   next values of RHO and DELTA.
        }
        case 680: {
            if (rho > rhoend) {
                delta = HALF * rho;
                ratio = rho / rhoend;
                if (ratio <= 16.) {
                    rho = rhoend;
                } else if (ratio <= 250.) {
                    rho = Math.sqrt(ratio) * rhoend;
                } else {
                    rho = TENTH * rho;
                }
                delta = Math.max(delta,rho);
                ntrits = 0;
                nfsav = nf.value;
                state = 60; break;
            }

            // Return from the calculation, after another Newton-Raphson step, if
            //   it is too short to have been tried before.

            if (ntrits == -1) {
                state = 360; break;
            }
        }
        case 720: {
            if (fval[kopt.value] <= fsave) {
                for (int i = 0; i < n; i++) {
                    // Computing MIN
                    // Computing MAX
                    double d__3 = xl[i];
                    double d__4 = xbase[i] + xopt[i];
                    double d__1 = Math.max(d__3,d__4);
                    double d__2 = xu[i];
                    x[i] = Math.min(d__1,d__2);
                    if (xopt[i] == sl[i]) {
                        x[i] = xl[i];
                    }
                    if (xopt[i] == su[i]) {
                        x[i] = xu[i];
                    }
                }
                f = fval[kopt.value];
            }
            return f;
        }}
    } // bobyqb

    // ----------------------------------------------------------------------------------------

    /**
     *     The arguments N, NPT, XPT, XOPT, BMAT, ZMAT, NDIM, SL and SU all have
     *       the same meanings as the corresponding arguments of BOBYQB.
     *     KOPT is the index of the optimal interpolation point.
     *     KNEW is the index of the interpolation point that is going to be moved.
     *     ADELT is the current trust region bound.
     *     XNEW will be set to a suitable new position for the interpolation point
     *       XPT(KNEW,.). Specifically, it satisfies the SL, SU and trust region
     *       bounds and it should provide a large denominator in the next call of
     *       UPDATE. The step XNEW-XOPT from XOPT is restricted to moves along the
     *       straight lines through XOPT and another interpolation point.
     *     XALT also provides a large value of the modulus of the KNEW-th Lagrange
     *       function subject to the constraints that have been mentioned, its main
     *       difference from XNEW being that XALT-XOPT is a constrained version of
     *       the Cauchy step within the trust region. An exception is that XALT is
     *       not calculated if all components of GLAG (see below) are ZERO.
     *     ALPHA will be set to the KNEW-th diagonal element of the H matrix.
     *     CAUCHY will be set to the square of the KNEW-th Lagrange function at
     *       the step XALT-XOPT from XOPT for the vector XALT that is returned,
     *       except that CAUCHY is set to ZERO if XALT is not calculated.
     *     GLAG is a working space vector of length N for the gradient of the
     *       KNEW-th Lagrange function at XOPT.
     *     HCOL is a working space vector of length NPT for the second derivative
     *       coefficients of the KNEW-th Lagrange function.
     *     W is a working space vector of length 2N that is going to hold the
     *       constrained Cauchy step from XOPT of the Lagrange function, followed
     *       by the downhill version of XALT when the uphill step is calculated.
     *
     *     Set the first NPT components of W to the leading elements of the
     *     KNEW-th column of the H matrix.
     * @param kopt
     * @param knew
     * @param adelt
     * @param alpha
     * @param cauchy
     */
    private void altmov(
            int kopt,
            int knew,
            double adelt,
            DoubleRef alpha,
            DoubleRef cauchy
    ) {
        double const__ = ONE + Math.sqrt(2.);
        double vlg = 0;
        double step = 0;
        double stpsav = 0;
        double bigstp = 0;
        double wsqsav = 0;
        double csave = 0;
        int ksav = 0;
        int ibdsav = 0;
        int isbd = 0;
        int ilbd = 0;
        int iubd = 0;
        
        for (int k = 0; k < npt; k++) {
            hcol[k] = ZERO;
        }
        int i1 = npt - n - 1;
        for (int j = 0; j < i1; j++) {
            double temp = zmat[knew + j * npt];
            for (int k = 0; k < npt; k++) {
                hcol[k] += temp * zmat[k + j * npt];
            }
        }
        alpha.value = hcol[knew];
        double ha = HALF * alpha.value;

        // Calculate the gradient of the KNEW-th Lagrange function at XOPT.

        for (int i = 0; i < n; i++) {
            glag[i] = bmat[knew + i * ndim];
        }
        for (int k = 0; k < npt; k++) {
            double temp = ZERO;
            for (int j = 0; j < n; j++) {
                temp += xpt[k + j * npt] * xopt[j];
            }
            temp = hcol[k] * temp;
            for (int i = 0; i < n; i++) {
                glag[i] += temp * xpt[k + i * npt];
            }
        }

        // Search for a large denominator along the straight lines through XOPT
        // and another interpolation point. SLBD and SUBD will be lower and upper
        // bounds on the step along each of these lines in turn. PREDSQ will be
        // set to the square of the predicted denominator for each line. PRESAV
        // will be set to the largest admissible value of PREDSQ that occurs.

        double presav = ZERO;
        for (int k = 0; k < npt; k++) {
            if (k == kopt) {
                continue;
            }
            double dderiv = ZERO;
            double distsq = ZERO;
            for (int i = 0; i < n; i++) {
                double temp = xpt[k + i * npt] - xopt[i];
                dderiv += glag[i] * temp;
                distsq += temp * temp;
            }
            double subd = adelt / Math.sqrt(distsq);
            double slbd = -subd;
            ilbd = 0;
            iubd = 0;
            double sumin = Math.min(ONE,subd);

            // Revise SLBD and SUBD if necessary because of the bounds in SL and SU.

            for (int i = 0; i < n; i++) {
                double temp = xpt[k + i * npt] - xopt[i];
                if (temp > ZERO) {
                    if (slbd * temp < sl[i] - xopt[i]) {
                        slbd = (sl[i] - xopt[i]) / temp;
                        ilbd = -i-1;
                    }
                    if (subd * temp > su[i] - xopt[i]) {
                        // Computing MAX
                        double d__1 = sumin;
                        double d__2 = (su[i] - xopt[i]) / temp;
                        subd = Math.max(d__1,d__2);
                        iubd = i+1;
                    }
                } else if (temp < ZERO) {
                    if (slbd * temp > su[i] - xopt[i]) {
                        slbd = (su[i] - xopt[i]) / temp;
                        ilbd = i+1;
                    }
                    if (subd * temp < sl[i] - xopt[i]) {
                        // Computing MAX
                        double d__1 = sumin;
                        double d__2 = (sl[i] - xopt[i]) / temp;
                        subd = Math.max(d__1,d__2);
                        iubd = -i-1;
                    }
                }
            }

            // Seek a large modulus of the KNEW-th Lagrange function when the index
            // of the other interpolation point on the line through XOPT is KNEW.

            if (k == knew) {
                double diff = dderiv - ONE;
                step = slbd;
                vlg = slbd * (dderiv - slbd * diff);
                isbd = ilbd;
                double temp = subd * (dderiv - subd * diff);
                if (Math.abs(temp) > Math.abs(vlg)) {
                    step = subd;
                    vlg = temp;
                    isbd = iubd;
                }
                double tempd = HALF * dderiv;
                double tempa = tempd - diff * slbd;
                double tempb = tempd - diff * subd;
                if (tempa * tempb < ZERO) {
                    temp = tempd * tempd / diff;
                    if (Math.abs(temp) > Math.abs(vlg)) {
                        step = tempd / diff;
                        vlg = temp;
                        isbd = 0;
                    }
                }

                // Search along each of the other lines through XOPT and another point.

            } else {
                step = slbd;
                vlg = slbd * (ONE - slbd);
                isbd = ilbd;
                double temp = subd * (ONE - subd);
                if (Math.abs(temp) > Math.abs(vlg)) {
                    step = subd;
                    vlg = temp;
                    isbd = iubd;
                }
                if (subd > HALF) {
                    if (Math.abs(vlg) < .25) {
                        step = HALF;
                        vlg = .25;
                        isbd = 0;
                    }
                }
                vlg *= dderiv;
            }

            // Calculate PREDSQ for the current line search and maintain PRESAV.

            double temp = step * (ONE - step) * distsq;
            double predsq = vlg * vlg * (vlg * vlg + ha * temp * temp);
            if (predsq > presav) {
                presav = predsq;
                ksav = k;
                stpsav = step;
                ibdsav = isbd;
            }
        }

        // Construct XNEW in a way that satisfies the bound constraints exactly.

        for (int i = 0; i < n; i++) {
            double temp = xopt[i] + stpsav * (xpt[ksav + i * npt] - xopt[i]);
            // Computing MAX
            // Computing MIN
            double d__3 = su[i];
            double d__1 = sl[i];
            double d__2 = Math.min(d__3,temp);
            xnew[i] = Math.max(d__1,d__2);
        }
        if (ibdsav < 0) {
            xnew[-ibdsav-1] = sl[-ibdsav-1];
        }
        if (ibdsav > 0) {
            xnew[ibdsav-1] =  su[ibdsav-1];
        }

        // Prepare for the iterative method that assembles the constrained Cauchy
        // step in W. The sum of squares of the fixed components of W is formed in
        // WFIXSQ, and the free components of W are set to BIGSTP.

        bigstp = adelt + adelt;
        double iflag = 0;

        L100: for(;;) {
            double wfixsq = ZERO;
            double ggfree = ZERO;
            for (int i = 0; i < n; i++) {
                w[i] = ZERO;
                // Computing MIN
                double d__1 = xopt[i] - sl[i];
                double d__2 = glag[i];
                double tempa = Math.min(d__1,d__2);
                // Computing MAX
                d__1 = xopt[i] - su[i];
                d__2 = glag[i];
                double tempb = Math.max(d__1,d__2);
                if (tempa > ZERO || tempb < ZERO) {
                    w[i] = bigstp;
                    // Computing 2nd power
                    d__1 = glag[i];
                    ggfree += d__1 * d__1;
                }
            }
            if (ggfree == ZERO) {
                cauchy.value = ZERO;
                return;
            }

            // Investigate whether more components of W can be fixed.
            L120: {
                double temp = adelt * adelt - wfixsq;
                if (temp > ZERO) {
                    wsqsav = wfixsq;
                    step = Math.sqrt(temp / ggfree);
                    ggfree = ZERO;
                    for (int i = 0; i < n; i++) {
                        if (w[i] == bigstp) {
                            temp = xopt[i] - step * glag[i];
                            if (temp <= sl[i]) {
                                w[i] = sl[i] - xopt[i];
                                // Computing 2nd power
                                double d__1 = w[i];
                                wfixsq += d__1 * d__1;
                            } else if (temp >= su[i]) {
                                w[i] = su[i] - xopt[i];
                                // Computing 2nd power
                                double d__1 = w[i];
                                wfixsq += d__1 * d__1;
                            } else {
                                // Computing 2nd power
                                double d__1 = glag[i];
                                ggfree += d__1 * d__1;
                            }
                        }
                    }
                    if (!(wfixsq > wsqsav && ggfree > ZERO)) {
                        break L120;
                    }
                }} // end L120

            // Set the remaining free components of W and all components of XALT,
            // except that W may be scaled later.

            double gw = ZERO;
            for (int i = 0; i < n; i++) {
                if (w[i] == bigstp) {
                    w[i] = -step * glag[i];
                    // Computing MAX
                    // Computing MIN
                    double d__3 = su[i];
                    double d__4 = xopt[i] + w[i];
                    double d__1 = sl[i];
                    double d__2 = Math.min(d__3,d__4);
                    xalt[i] = Math.max(d__1,d__2);
                } else if (w[i] == ZERO) {
                    xalt[i] = xopt[i];
                } else if (glag[i] > ZERO) {
                    xalt[i] = sl[i];
                } else {
                    xalt[i] = su[i];
                }
                gw += glag[i] * w[i];
            }

            // Set CURV to the curvature of the KNEW-th Lagrange function along W.
            // Scale W by a factor less than one if that can reduce the modulus of
            // the Lagrange function at XOPT+W. Set CAUCHY to the final value of
            // the square of this function.

            double curv = ZERO;
            for (int k = 0; k < npt; k++) {
                double temp = ZERO;
                for (int j = 0; j < n; j++) {
                    temp += xpt[k + j * npt] * w[j];
                }
                curv += hcol[k] * temp * temp;
            }
            if (iflag == 1) {
                curv = -curv;
            }
            if (curv > -gw && curv < -const__ * gw) {
                double scale = -gw / curv;
                for (int i = 0; i < n; i++) {
                    double temp = xopt[i] + scale * w[i];
                    // Computing MAX
                    // Computing MIN
                    double d__3 = su[i];
                    double d__1 = sl[i];
                    double d__2 = Math.min(d__3,temp);
                    xalt[i] = Math.max(d__1,d__2);
                }
                // Computing 2nd power
                double d__1 = HALF * gw * scale;
                cauchy.value = d__1 * d__1;
            } else {
                // Computing 2nd power
                double d__1 = gw + HALF * curv;
                cauchy.value = d__1 * d__1;
            }

            // If IFLAG is ZERO, then XALT is calculated as before after reversing
            // the sign of GLAG. Thus TWO XALT vectors become available. The one that
            // is chosen is the one that gives the larger value of CAUCHY.

            if (iflag == 0) {
                for (int i = 0; i < n; i++) {
                    glag[i] = -glag[i];
                    w[n + i] = xalt[i];
                }
                csave = cauchy.value;
                iflag = 1;
            } else {
                break L100;
            }} // end L100
        if (csave > cauchy.value) {
            for (int i = 0; i < n; i++) {
                xalt[i] = w[n + i];
            }
            cauchy.value = csave;
        }
    } // altmov

    // ----------------------------------------------------------------------------------------

    /**
     *     SUBROUTINE PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
     *     BMAT and ZMAT for the first iteration, and it maintains the values of
     *     NF and KOPT. The vector X is also changed by PRELIM.
     *
     *     The arguments N, NPT, X, XL, XU, RHOBEG, IPRINT and MAXFUN are the
     *       same as the corresponding arguments in SUBROUTINE BOBYQA.
     *     The arguments XBASE, XPT, FVAL, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU
     *       are the same as the corresponding arguments in BOBYQB, the elements
     *       of SL and SU being set in BOBYQA.
     *     GOPT is usually the gradient of the quadratic model at XOPT+XBASE, but
     *       it is set by PRELIM to the gradient of the quadratic model at XBASE.
     *       If XOPT is nonZERO, BOBYQB will change it to its usual value later.
     *     NF is maintaned as the number of calls of CALFUN so far.
     *     KOPT will be such that the least calculated value of F so far is at
     *       the point XPT(KOPT,.)+XBASE in the space of the variables.
     *
     * @param maxfun
     * @param nf
     * @param kopt
     */
    private void prelim(
            int maxfun,
            IntRef nf,
            IntRef kopt
    ) {
        double rhosq = rhobeg * rhobeg;
        double recip = ONE / rhosq;
        double fbeg = 0;
        double stepa = 0;
        double stepb = 0;
        int jpt = 0;
        int ipt = 0;

        // Set XBASE to the initial vector of variables, and set the initial
        // elements of XPT, BMAT, HQ, PQ and ZMAT to ZERO.

        for (int j = 0; j < n; j++) {
            xbase[j] = x[j];
            for (int k = 0; k < npt; k++) {
                xpt[k + j * npt] = ZERO;
            }
            for (int i = 0; i < ndim; i++) {
                bmat[i + j * ndim] = ZERO;
            }
        }
        int i2 = n * np / 2;
        for (int ih = 0; ih < i2; ih++) {
            hq[ih] = ZERO;
        }
        for (int k = 0; k < npt; k++) {
            pq[k] = ZERO;
            int i1 = npt - np;
            for (int j = 0; j < i1; j++) {
                zmat[k + j * npt] = ZERO;
            }
        }

        // Begin the initialization procedure. NF becomes the number
        // of function values so far. The coordinates of the displacement of the
        // next initial interpolation point from XBASE are set in XPT(NF+1,.).

        nf.value = 0;
        do {
            int nfm = nf.value;
            int nfx = nf.value - n;
            int nfmm = nfm - 1;
            int nfxm = nfx - 1;
            nf.value++;
            if (nfm <= n << 1) {
                if (nfm >= 1 && nfm <= n) {
                    stepa = rhobeg;
                    if (su[nfmm] == ZERO) {
                        stepa = -stepa;
                    }
                    xpt[nfm + nfmm * npt] = stepa;
                } else if (nfm > n) {
                    stepa = xpt[nfx + nfxm * npt];
                    stepb = -rhobeg;
                    if (sl[nfxm] == ZERO) {
                        // Computing MIN
                        double d__1 = TWO * rhobeg;
                        double d__2 = su[nfxm];
                        stepb = Math.min(d__1,d__2);
                    }
                    if (su[nfxm] == ZERO) {
                        // Computing MAX
                        double d__1 = -TWO * rhobeg;
                        double d__2 = sl[nfxm];
                        stepb = Math.max(d__1,d__2);
                    }
                    xpt[nfm + nfxm * npt] = stepb;
                }
            } else {
                int itemp = (nfm - np) / n;
                jpt = nfm - itemp * n - n;
                ipt = jpt + itemp;
                if (ipt > n) {
                    itemp = jpt;
                    jpt = ipt - n;
                    ipt = itemp;
                }
                xpt[nfm + (ipt-1) * npt] = xpt[ipt + (ipt-1) * npt];
                xpt[nfm + (jpt-1) * npt] = xpt[jpt + (jpt-1) * npt];
            }

            // Calculate the next value of F. The least function value so far and
            // its index are required.

            for (int j = 0; j < n; j++) {
                // Computing MIN
                // Computing MAX
                double d__3 = xl[j];
                double d__4 = xbase[j] + xpt[nfm + j * npt];
                double d__1 = Math.max(d__3,d__4);
                double d__2 = xu[j];
                x[j] = Math.min(d__1,d__2);
                if (xpt[nfm + j * npt] == sl[j]) {
                    x[j] = xl[j];
                }
                if (xpt[nfm + j * npt] == su[j]) {
                    x[j] = xu[j];
                }
            }
            double f = computeObjectiveValue(x);
            if (!isMinimize)
                f = -f;
            fval[nfm] = f;
            if (nfm == 0) {
                fbeg = f;
                kopt.value = 0;
            } else if (f < fval[kopt.value]) {
                kopt.value = nfm;
            }

            // Set the nonZERO initial elements of BMAT and the quadratic model in the
            // cases when NF is at most 2*N+1. If NF exceeds N+1, then the positions
            // of the NF-th and (NF-N)-th interpolation points may be switched, in
            // order that the function value at the first of them contributes to the
            // off-diagonal second derivative terms of the initial quadratic model.

            if (nf.value <= (n << 1) + 1) {
                if (nf.value >= 2 && nf.value <= n + 1) {
                    gopt[nfmm] = (f - fbeg) / stepa;
                    if (npt < nf.value + n) {
                        bmat[nfmm * ndim] = -ONE / stepa;
                        bmat[nfm + nfmm * ndim] = ONE / stepa;
                        bmat[npt + nfmm + nfmm * ndim] = -HALF * rhosq;
                    }
                } else if (nf.value >= n + 2) {
                    int ih = nfx * (nfx + 1) / 2 - 1;
                    double temp = (f - fbeg) / stepb;
                    double diff = stepb - stepa;
                    hq[ih] = TWO * (temp - gopt[nfxm]) / diff;
                    gopt[nfxm] = (gopt[nfxm] * stepb - temp * stepa) / diff;
                    if (stepa * stepb < ZERO) {
                        if (f < fval[nfm - n]) {
                            fval[nfm] = fval[nfm - n];
                            fval[nfm - n] = f;
                            if (kopt.value == nfm) {
                                kopt.value = nfm - n;
                            }
                            xpt[nfm - n + nfxm * npt] = stepb;
                            xpt[nfm + nfxm * npt] = stepa;
                        }
                    }
                    bmat[nfxm * ndim] = -(stepa + stepb) / (stepa * stepb);
                    bmat[nfm + nfxm * ndim] = -HALF /
                            xpt[nfm - n + nfxm * npt];
                    bmat[nfm - n + nfxm * ndim] = -bmat[nfxm * ndim] -
                            bmat[nfm + nfxm * ndim];
                    zmat[nfxm * npt] = Math.sqrt(TWO) / (stepa * stepb);
                    zmat[nfm + nfxm * npt] = Math.sqrt(HALF) / rhosq;
                    zmat[nfm - n + nfxm * npt] = -zmat[nfxm * npt] -
                            zmat[nfm + nfxm * npt];
                }

                // Set the off-diagonal second derivatives of the Lagrange functions and
                // the initial quadratic model.

            } else {
                int ih = ipt * (ipt - 1) / 2 + jpt - 1;
                zmat[nfxm * npt] = recip;
                zmat[nfm + nfxm * npt] = recip;
                zmat[ipt + nfxm * npt] = -recip;
                zmat[jpt + nfxm * npt] = -recip;
                double temp = xpt[nfm + (ipt-1) * npt] * xpt[nfm + (jpt-1) * npt];
                hq[ih] = (fbeg - fval[ipt] - fval[jpt] + f) / temp;
            }
        } while (nf.value < npt && nf.value < maxfun);
    } // prelim

    // ----------------------------------------------------------------------------------------

    /**
     *     The first NDIM+NPT elements of the array W are used for working space.
     *     The final elements of BMAT and ZMAT are set in a well-conditioned way
     *       to the values that are appropriate for the new interpolation points.
     *     The elements of GOPT, HQ and PQ are also revised to the values that are
     *       appropriate to the final quadratic model.
     *
     *     The arguments N, NPT, XL, XU, IPRINT, MAXFUN, XBASE, XPT, FVAL, XOPT,
     *       GOPT, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU have the same meanings as
     *       the corresponding arguments of BOBYQB on the entry to RESCUE.
     *     NF is maintained as the number of calls of CALFUN so far, except that
     *       NF is set to -1 if the value of MAXFUN prevents further progress.
     *     KOPT is maintained so that FVAL(KOPT) is the least calculated function
     *       value. Its correct value must be given on entry. It is updated if a
     *       new least function value is found, but the corresponding changes to
     *       XOPT and GOPT have to be made later by the calling program.
     *     DELTA is the current trust region radius.
     *     VLAG is a working space vector that will be used for the values of the
     *       provisional Lagrange functions at each of the interpolation points.
     *       They are part of a product that requires VLAG to be of length NDIM.
     *     PTSAUX is also a working space array. For J=1,2,...,N, PTSAUX(1,J) and
     *       PTSAUX(2,J) specify the two positions of provisional interpolation
     *       points when a nonzero step is taken along e_J (the J-th coordinate
     *       direction) through XBASE+XOPT, as specified below. Usually these
     *       steps have length DELTA, but other lengths are chosen if necessary
     *       in order to satisfy the given bounds on the variables.
     *     PTSID is also a working space array. It has NPT components that denote
     *       provisional new positions of the original interpolation points, in
     *       case changes are needed to restore the linear independence of the
     *       interpolation conditions. The K-th point is a candidate for change
     *       if and only if PTSID(K) is nonzero. In this case let p and q be the
     *       int parts of PTSID(K) and (PTSID(K)-p) multiplied by N+1. If p
     *       and q are both positive, the step from XBASE+XOPT to the new K-th
     *       interpolation point is PTSAUX(1,p)*e_p + PTSAUX(1,q)*e_q. Otherwise
     *       the step is PTSAUX(1,p)*e_p or PTSAUX(2,q)*e_q in the cases q=0 or
     *       p=0, respectively.
     * @param nf
     * @param delta
     * @param kopt
     */
    private void rescue (
            IntRef nf,
            double delta,
            IntRef kopt
     ) {
        int maxfun = getMaxEvaluations();
    	double[] w = new double[ndim+npt];
        int np = n + 1;
        double sfrac = HALF / (double) np;

        // Shift the interpolation points so that XOPT becomes the origin, and set
        // the elements of ZMAT to ZERO. The value of SUMPQ is required in the
        // updating of HQ below. The squares of the distances from XOPT to the
        // other interpolation points are set at the end of W. Increments of WINC
        // may be added later to these squares to balance the consideration of
        // the choice of point that is going to become current.

        double sumpq = ZERO;
        double winc = ZERO;
        double distsq;
        double beta = 0;
        double denom = 0;
        
        for (int k = 0; k < npt; k++) {
            distsq = ZERO;
            for (int j = 0; j < n; j++) {
                xpt[k + j * npt] -= xopt[j];
                // Computing 2nd power
                double d__1 = xpt[k + j * npt];
                distsq += d__1 * d__1;
            }
            sumpq += pq[k];
            w[ndim + k] = distsq;
            winc = Math.max(winc,distsq);
            for (int j = 0; j < nptm; j++) {
                zmat[k + j * npt] = ZERO;
            }
        }

        // Update HQ so that HQ and PQ define the second derivatives of the model
        // after XBASE has been shifted to the trust region centre.

        int ih = 0;
        for (int j = 0; j < n; j++) {
            w[j] = HALF * sumpq * xopt[j];
            for (int k = 0; k < npt; k++) {
                w[j] += pq[k] * xpt[k + j * npt];
            }
            for (int i = 0; i <= j; i++) {
                hq[ih] += w[i] * xopt[j] + w[j] * xopt[i];
                ih++;
            }
        }

        // Shift XBASE, SL, SU and XOPT. Set the elements of BMAT to ZERO, and
        // also set the elements of PTSAUX.

        for (int j = 0; j < n; j++) {
            xbase[j] = xbase[j] + xopt[j];
            sl[j] -= xopt[j];
            su[j] -= xopt[j];
            xopt[j] = ZERO;
            // Computing MIN
            double d__1 = delta;
            double d__2 = su[j];
            ptsaux[(j << 1)] = Math.min(d__1,d__2);
            // Computing MAX
            d__1 = -delta;
            d__2 = sl[j];
            ptsaux[(j << 1) + 1] = Math.max(d__1,d__2);
            if (ptsaux[(j << 1)] + ptsaux[(j << 1) + 1] < ZERO) {
                double temp = ptsaux[(j << 1)];
                ptsaux[(j << 1)] = ptsaux[(j << 1) + 1];
                ptsaux[(j << 1) + 1] = temp;
            }
            d__2 = ptsaux[(j << 1) + 1];
            d__1 = ptsaux[(j << 1)];
            if (Math.abs(d__2) < HALF * Math.abs(d__1)) {
                ptsaux[(j << 1) + 1] = HALF * ptsaux[(j << 1)];
            }
            for (int i = 0; i < ndim; i++) {
                bmat[i + j * ndim] = ZERO;
            }
        }
        double fbase = fval[kopt.value];

        // Set the identifiers of the artificial interpolation points that are
        // along a coordinate direction from XOPT, and set the corresponding
        // nonzero elements of BMAT and ZMAT.

        ptsid[0] = sfrac;
        for (int j = 0; j < n; j++) {
            int jp = j + 1;
            int jpn = jp + n;
            ptsid[jp] = 1.0 + j + sfrac;
            if (jpn < npt) {
                ptsid[jpn] = (1.0+j) / np + sfrac;
                double temp = ONE / (ptsaux[(j << 1)] - ptsaux[(j << 1) + 1]);
                bmat[jp + j * ndim] = -temp + ONE / ptsaux[(j << 1)];
                bmat[jpn + j * ndim] = temp + ONE / ptsaux[(j << 1) + 1];
                bmat[j * ndim] = -bmat[jp + j * ndim] - bmat[jpn +
                        j * ndim];
                double d__1 = ptsaux[(j << 1)] * ptsaux[(j << 1) + 1];
                zmat[j * npt] = Math.sqrt(2.) / Math.abs(d__1);
                zmat[jp + j * npt] = zmat[j * npt] *
                        ptsaux[(j << 1) + 1] * temp;
                zmat[jpn + j * npt] = -zmat[j * npt] *
                        ptsaux[(j << 1)] * temp;
            } else {
                bmat[j * ndim] = -ONE / ptsaux[(j << 1)];
                bmat[jp + j * ndim] = ONE / ptsaux[(j << 1)];
                // Computing 2nd power
                double d__1 = ptsaux[(j << 1)];
                bmat[j + npt + j * ndim] = -HALF * (d__1 * d__1);
            }
        }

        // Set any remaining identifiers with their nonzero elements of ZMAT.

        if (npt >= n + np) {
             for (int k = np << 1; k <= npt; k++) {
                int iw = (int) (((double) (k - np) - HALF) / (double) n);
                int ip = k - np - iw * n;
                int iq = ip + iw;
                if (iq > n) {
                    iq -= n;
                }
                ptsid[k-1] = (double) ip + (double) iq / (double) np +
                        sfrac;
                double temp = ONE / (ptsaux[(ip << 1)] * ptsaux[(iq << 1)]);
                zmat[(k - np - 1) * npt] =  temp;
                zmat[ip + (k - np - 1) * npt] = -temp;
                zmat[iq + (k - np - 1) * npt] = -temp;
                zmat[k-1 + (k - np - 1) * npt] = temp;
            }
        }
        int nrem = npt;
        int kold = 0;
        int knew = kopt.value;

        // Reorder the provisional points in the way that exchanges PTSID(KOLD)
        // with PTSID(KNEW).

        int state = 80;
        for(;;) switch (state) {
        case 80: {
            for (int j = 0; j < n; j++) {
                double temp = bmat[kold + j * ndim];
                bmat[kold + j * ndim] = bmat[knew + j * ndim];
                bmat[knew + j * ndim] = temp;
            }
            for (int j = 0; j < nptm; j++) {
                double temp = zmat[kold + j * npt];
                zmat[kold + j * npt] = zmat[knew + j * npt];
                zmat[knew + j * npt] = temp;
            }
            ptsid[kold] = ptsid[knew];
            ptsid[knew] = ZERO;
            w[ndim + knew] = ZERO;
            --nrem;
            if (knew != kopt.value) {
                double temp = vlag[kold];
                vlag[kold] = vlag[knew];
                vlag[knew] = temp;

                // Update the BMAT and ZMAT matrices so that the status of the KNEW-th
                // interpolation point can be changed from provisional to original. The
                // branch to label 350 occurs if all the original points are reinstated.
                // The nonnegative values of W(NDIM+K) are required in the search below.

                update(beta, denom, knew);

                if (nrem == 0) {
                    return;
                }
                for (int k = 0; k < npt; k++) {
                    double d__1 = w[ndim + k];
                    w[ndim + k] = Math.abs(d__1);
                }
            }

            // Pick the index KNEW of an original interpolation point that has not
            // yet replaced one of the provisional interpolation points, giving
            // attention to the closeness to XOPT and to previous tries with KNEW.
        }
        case 120: {
            double dsqmin = ZERO;
            for (int k = 0; k < npt; k++) {
                if (w[ndim + k] > ZERO) {
                    if (dsqmin == ZERO || w[ndim + k] < dsqmin) {
                        knew = k;
                        dsqmin = w[ndim + k];
                    }
                }
            }
            if (dsqmin == ZERO) {
                state = 260; break;
            }

            // Form the W-vector of the chosen original interpolation point.

            for (int j = 0; j < n; j++) {
                w[npt + j] = xpt[knew + j * npt];
            }
            for (int k = 0; k < npt; k++) {
                double sum = ZERO;
                if (k == kopt.value) {
                } else if (ptsid[k] == ZERO) {
                    for (int j = 0; j < n; j++) {
                        sum += w[npt + j] * xpt[k + j * npt];
                    }
                } else {
                    int ip = (int) ptsid[k];
                    if (ip > 0) {
                        sum = w[npt + ip - 1] * ptsaux[(ip-1) << 1];
                    }
                    int iq = (int) ((double) np * ptsid[k] - (double) (ip * np));
                    if (iq > 0) {
                        int iw = 0;
                        if (ip == 0) {
                            iw = 1;
                        }
                        sum += w[npt + iq - 1] * ptsaux[iw + ((iq-1) << 1)];
                    }
                }
                w[k] = HALF * sum * sum;
            }

            // Calculate VLAG and BETA for the required updating of the H matrix if
            // XPT(KNEW,.) is reinstated in the set of interpolation points.

            for (int k = 0; k < npt; k++) {
                double sum = ZERO;
                for (int j = 0; j < n; j++) {
                    sum += bmat[k + j * ndim] * w[npt + j];
                }
                vlag[k] = sum;
            }
            beta = ZERO;
            for (int j = 0; j < nptm; j++) {
                double sum = ZERO;
                for (int k = 0; k < npt; k++) {
                    sum += zmat[k + j * npt] * w[k];
                }
                beta -= sum * sum;
                for (int k = 0; k < npt; k++) {
                    vlag[k] += sum * zmat[k + j * npt];
                }
            }
            double bsum = ZERO;
            distsq = ZERO;
            for (int j = 0; j < n; j++) {
                double sum = ZERO;
                for (int k = 0; k < npt; k++) {
                    sum += bmat[k + j * ndim] * w[k];
                }
                int jp = j + npt;
                bsum += sum * w[jp];
                for (int ip = npt-1; ip < ndim; ip++) {
                    sum += bmat[ip + j * ndim] * w[ip];
                }
                bsum += sum * w[jp];
                vlag[jp] = sum;
                // Computing 2nd power
                double d__1 = xpt[knew + j * npt];
                distsq += d__1 * d__1;
            }
            beta = HALF * distsq * distsq + beta - bsum;
            vlag[kopt.value] += ONE;

            // KOLD is set to the index of the provisional interpolation point that is
            // going to be deleted to make way for the KNEW-th original interpolation
            // point. The choice of KOLD is governed by the avoidance of a small value
            // of the denominator in the updating calculation of UPDATE.

            denom = ZERO;
            double vlmxsq = ZERO;
            for (int k = 0; k < npt; k++) {
                if (ptsid[k] != ZERO) {
                    double hdiag = ZERO;
                     for (int j = 0; j < nptm; j++) {
                        // Computing 2nd power
                        double d__1 = zmat[k + j * npt];
                        hdiag += d__1 * d__1;
                    }
                    // Computing 2nd power
                    double d__1 = vlag[k];
                    double den = beta * hdiag + d__1 * d__1;
                    if (den > denom) {
                        kold = k;
                        denom = den;
                    }
                }
                // Computing MAX
                // Computing 2nd power
                double d__3 = vlag[k];
                double d__1 = vlmxsq;
                double d__2 = d__3 * d__3;
                vlmxsq = Math.max(d__1,d__2);
            }
            if (denom <= vlmxsq * .01) {
                w[ndim + knew] = -w[ndim + knew] - winc;
                state = 120; break;
            }
            state = 80; break;

            // When label 260 is reached, all the final positions of the interpolation
            // points have been chosen although any changes have not been included yet
            // in XPT. Also the final BMAT and ZMAT matrices are complete, but, apart
            // from the shift of XBASE, the updating of the quadratic model remains to
            // be done. The following cycle through the new interpolation points begins
            // by putting the new point in XPT(KPT,.) and by setting PQ(KPT) to ZERO,
            // except that a RETURN occurs if MAXFUN prohibits another value of F.

        }
        case 260: {
            for (int kpt = 0; kpt < npt; kpt++) {
                if (ptsid[kpt] == ZERO) {
                    continue;
                }
                if (nf.value >= maxfun) {
                    nf.value = -1;
                    return;
                }
                ih = 0;
                for (int j = 0; j < n; j++) {
                    w[j] = xpt[kpt + j * npt];
                    xpt[kpt + j * npt] = ZERO;
                    double temp = pq[kpt] * w[j];
                    for (int i = 0; i <= j; i++) {
                        hq[ih] += temp * w[i];
                        ih++;
                    }
                }
                pq[kpt] = ZERO;
                int ip = (int) ptsid[kpt];
                int iq = (int) ((double) np * ptsid[kpt] - (double) (ip * np));
                double xp = 0;
                double xq = 0;
                if (ip > 0) {
                    xp = ptsaux[(ip-1) << 1];
                    xpt[kpt + (ip-1) * npt] = xp;
                }
                if (iq > 0) {
                    xq = ptsaux[(iq-1) << 1];
                    if (ip == 0) {
                        xq = ptsaux[((iq-1) << 1) + 1];
                    }
                    xpt[kpt + (iq-1) * npt] = xq;
                }

                // Set VQUAD to the value of the current model at the new point.

                double vquad = fbase;
                int ihp = 0;
                if (ip > 0) {
                    ihp = (ip + ip * ip) / 2;
                    vquad += xp * (gopt[ip-1] + HALF * xp * hq[ihp-1]);
                }
                if (iq > 0) {
                    int ihq = (iq + iq * iq) / 2;
                    vquad += xq * (gopt[iq-1] + HALF * xq * hq[ihq-1]);
                    if (ip > 0) {
                        int iw = Math.max(ihp,ihq) - Math.abs(ip - iq);
                        vquad += xp * xq * hq[iw-1];
                    }
                }
                for (int k = 0; k < npt; k++) {
                    double temp = ZERO;
                    if (ip > 0) {
                        temp += xp * xpt[k + (ip-1) * npt];
                    }
                    if (iq > 0) {
                        temp += xq * xpt[k + (iq-1) * npt];
                    }
                    vquad += HALF * pq[k] * temp * temp;
                }

                // Calculate F at the new interpolation point, and set DIFF to the factor
                // that is going to multiply the KPT-th Lagrange function when the model
                // is updated to provide interpolation to the new function value.

                for (int i = 0; i < n; i++) {
                    // Computing MIN
                    // Computing MAX
                    double d__3 = xl[i];
                    double d__4 = xbase[i] + xpt[kpt + i * npt];
                    double d__1 = Math.max(d__3,d__4);
                    double d__2 = xu[i];
                    w[i] = Math.min(d__1,d__2);
                    if (xpt[kpt + i * npt] == sl[i]) {
                        w[i] = xl[i];
                    }
                    if (xpt[kpt + i * npt] == su[i]) {
                        w[i] = xu[i];
                    }
                }
                nf.value++;
                double f = computeObjectiveValue(Arrays.copyOf(w,n));
                if (!isMinimize)
                    f = -f;
                fval[kpt] = f;
                if (f < fval[kopt.value]) {
                    kopt.value = kpt;
                }
                double diff = f - vquad;

                // Update the quadratic model. The RETURN from the subroutine occurs when
                // all the new interpolation points are included in the model.

                for (int i = 0; i < n; i++) {
                    gopt[i] += diff * bmat[kpt + i * ndim];
                }
                for (int k = 0; k < npt; k++) {
                    double sum = ZERO;
                    for (int j = 0; j < nptm; j++) {
                        sum += zmat[k + j * npt] * zmat[kpt + j * npt];
                    }
                    double temp = diff * sum;
                    if (ptsid[k] == ZERO) {
                        pq[k] += temp;
                    } else {
                        ip = (int) ptsid[k];
                        iq = (int) ((double) np * ptsid[k] - (double) (ip * np));
                        int ihq = (iq * iq + iq) / 2;
                        if (ip == 0) {
                            // Computing 2nd power
                            double d__1 = ptsaux[((iq-1) << 1) + 1];
                            hq[ihq-1] += temp * (d__1 * d__1);
                        } else {
                            ihp = (ip * ip + ip) / 2;
                            // Computing 2nd power
                            double d__1 = ptsaux[(ip-1) << 1];
                            hq[ihp-1] += temp * (d__1 * d__1);
                            if (iq > 0) {
                                // Computing 2nd power
                                d__1 = ptsaux[(iq-1) << 1];
                                hq[ihq-1] += temp * (d__1 * d__1);
                                int iw = Math.max(ihp,ihq) - Math.abs(iq - ip);
                                hq[iw-1] += temp * ptsaux[(ip-1) << 1] * ptsaux[(iq-1) << 1];
                            }
                        }
                    }
                }
                ptsid[kpt] = ZERO;
            }
            return;
        }}
    } // rescue


    // ----------------------------------------------------------------------------------------

    /**
     *     A version of the truncated conjugate gradient is applied. If a line
     *     search is restricted by a constraint, then the procedure is restarted,
     *     the values of the variables that are at their bounds being fixed. If
     *     the trust region boundary is reached, then further changes may be made
     *     to D, each one being in the two dimensional space that is spanned
     *     by the current D and the gradient of Q at XOPT+D, staying on the trust
     *     region boundary. Termination occurs when the reduction in Q seems to
     *     be close to the greatest reduction that can be achieved.
     *     The arguments N, NPT, XPT, XOPT, GOPT, HQ, PQ, SL and SU have the same
     *       meanings as the corresponding arguments of BOBYQB.
     *     DELTA is the trust region radius for the present calculation, which
     *       seeks a small value of the quadratic model within distance DELTA of
     *       XOPT subject to the bounds on the variables.
     *     XNEW will be set to a new vector of variables that is approximately
     *       the one that minimizes the quadratic model within the trust region
     *       subject to the SL and SU constraints on the variables. It satisfies
     *       as equations the bounds that become active during the calculation.
     *     D is the calculated trial step from XOPT, generated iteratively from an
     *       initial value of ZERO. Thus XNEW is XOPT+D after the final iteration.
     *     GNEW holds the gradient of the quadratic model at XOPT+D. It is updated
     *       when D is updated.
     *     xbdi[ is a working space vector. For I=1,2,...,N, the element xbdi[(I) is
     *       set to -1.0, 0.0, or 1.0, the value being nonzero if and only if the
     *       I-th variable has become fixed at a bound, the bound being SL(I) or
     *       SU(I) in the case xbdi[(I)=-1.0 or xbdi[(I)=1.0, respectively. This
     *       information is accumulated during the construction of XNEW.
     *     The arrays S, HS and HRED are also used for working space. They hold the
     *       current search direction, and the changes in the gradient of Q along S
     *       and the reduced D, respectively, where the reduced D is the same as D,
     *       except that the components of the fixed variables are ZERO.
     *     DSQ will be set to the square of the length of XNEW-XOPT.
     *     CRVMIN is set to ZERO if D reaches the trust region boundary. Otherwise
     *       it is set to the least curvature of H that occurs in the conjugate
     *       gradient searches that are not restricted by any constraints. The
     *       value CRVMIN=-1.0D0 is set, however, if all of these searches are
     *       constrained.
     * @param delta
     * @param dsq
     * @param crvmin
     */
    private void trsbox(
            double delta,
            DoubleRef dsq,
            DoubleRef crvmin
    ) {

        // The sign of GOPT(I) gives the sign of the change to the I-th variable
        // that will reduce Q from its value at XOPT. Thus xbdi[(I) shows whether
        // or not to fix the I-th variable at one of its bounds initially, with
        // NACT being set to the number of fixed variables. D and GNEW are also
        // set for the first iteration. DELSQ is the upper bound on the sum of
        // squares of the free variables. QRED is the reduction in Q so far.

        int iterc = 0;
        int nact = 0;
        for (int i = 0; i < n; i++) {
            xbdi[i] = ZERO;
            if (xopt[i] <= sl[i]) {
                if (gopt[i] >= ZERO) {
                    xbdi[i] = ONEMIN;
                }
            } else if (xopt[i] >= su[i]) {
                if (gopt[i] <= ZERO) {
                    xbdi[i] = ONE;
                }
            }
            if (xbdi[i] != ZERO) {
                ++nact;
            }
            d__[i] = ZERO;
            gnew[i] = gopt[i];
        }
        double delsq = delta * delta;
        double qred = ZERO;
        double beta = ZERO;
        double dredsq = ZERO;
        double dredg = ZERO;
        double gredsq = ZERO;
        double ggsav = ZERO;
        double stepsq = ZERO;
        double blen = ZERO;
        double stplen = ZERO;
        double xsav = ZERO;
        double angbd = ZERO;
        double sredg = ZERO;
        double rdprev = ZERO;
        double rdnext = ZERO;
        double angt = ZERO;
        int iact = -1;
        int itermax = 0;
        int itcsav = 0;
        crvmin.value = ONEMIN;
        
        // Set the next search direction of the conjugate gradient method. It is
        // the steepest descent direction initially and when the iterations are
        // restarted because a variable has just been fixed by a bound, and of
        // course the components of the fixed variables are ZERO. ITERMAX is an
        // upper bound on the indices of the conjugate gradient iterations.

        int state = 20;
        for(;;) L200 : switch (state) {

        case 20: {
            beta = ZERO;
        }
        case 30: {
            stepsq = ZERO;
            for (int i = 0; i < n; i++) {
                if (xbdi[i] != ZERO) {
                    s[i] = ZERO;
                } else if (beta == ZERO) {
                    s[i] = -gnew[i];
                } else {
                    s[i] = beta * s[i] - gnew[i];
                }
                // Computing 2nd power
                double d__1 = s[i];
                stepsq += d__1 * d__1;
            }
            if (stepsq == ZERO) {
                state = 190; break;
            }
            if (beta == ZERO) {
                gredsq = stepsq;
                itermax = iterc + n - nact;
            }
            if (gredsq * delsq <= qred * 1e-4 * qred) {
                state = 190; break;
            }

            // Multiply the search direction by the second derivative matrix of Q and
            // calculate some scalars for the choice of steplength. Then set BLEN to
            // the length of the the step to the trust region boundary and STPLEN to
            // the steplength, ignoring the simple bounds.

            state = 210; break;
        }
        case 50: {
            double resid = delsq;
            double ds = ZERO;
            double shs = ZERO;
            for (int i = 0; i < n; i++) {
                if (xbdi[i] == ZERO) {
                    // Computing 2nd power
                    double d__1 = d__[i];
                    resid -= d__1 * d__1;
                    ds += s[i] * d__[i];
                    shs += s[i] * hs[i];
                }
            }
            if (resid <= ZERO) {
                state = 90; break;
            }
            double temp = Math.sqrt(stepsq * resid + ds * ds);
            if (ds < ZERO) {
                blen = (temp - ds) / stepsq;
            } else {
                blen = resid / (temp + ds);
            }
            stplen = blen;
            if (shs > ZERO) {
                // Computing MIN
                double d__1 = blen;
                double d__2 = gredsq / shs;
                stplen = Math.min(d__1,d__2);
            }

            // Reduce STPLEN if necessary in order to preserve the simple bounds,
            // letting IACT be the index of the new constrained variable.

            iact = -1;
            for (int i = 0; i < n; i++) {
                if (s[i] != ZERO) {
                    double xsum = xopt[i] + d__[i];
                    if (s[i] > ZERO) {
                        temp = (su[i] - xsum) / s[i];
                    } else {
                        temp = (sl[i] - xsum) / s[i];
                    }
                    if (temp < stplen) {
                        stplen = temp;
                        iact = i;
                    }
                }
            }

            // Update CRVMIN, GNEW and D. Set SDEC to the decrease that occurs in Q.

            double sdec = ZERO;
            if (stplen > ZERO) {
                ++iterc;
                temp = shs / stepsq;
                if (iact < 0 && temp > ZERO) {
                    crvmin.value = Math.min(crvmin.value,temp);
                    if (crvmin.value == ONEMIN) {
                        crvmin.value = temp;
                    }
                }
                ggsav = gredsq;
                gredsq = ZERO;
                for (int i = 0; i < n; i++) {
                    gnew[i] = gnew[i] + stplen * hs[i];
                    if (xbdi[i] == ZERO) {
                        // Computing 2nd power
                        double d__1 = gnew[i];
                        gredsq += d__1 * d__1;
                    }
                    d__[i] += stplen * s[i];
                }
                // Computing MAX
                double d__1 = stplen * (ggsav - HALF * stplen * shs);
                sdec = Math.max(d__1,ZERO);
                qred += sdec;
            }

            // Restart the conjugate gradient method if it has hit a new bound.

            if (iact >= 0) {
                ++nact;
                xbdi[iact] = ONE;
                if (s[iact] < ZERO) {
                    xbdi[iact] = ONEMIN;
                }
                // Computing 2nd power
                double d__1 = d__[iact];
                delsq -= d__1 * d__1;
                if (delsq <= ZERO) {
                    state = 190; break;
                }
                state = 20; break;
            }

            // If STPLEN is less than BLEN, then either apply another conjugate
            // gradient iteration or RETURN.

            if (stplen < blen) {
                if (iterc == itermax) {
                    state = 190; break;
                }
                if (sdec <= qred * .01) {
                    state = 190; break;
                }
                beta = gredsq / ggsav;
                state = 30; break;
            }
        }
        case 90: {
            crvmin.value = ZERO;

            // Prepare for the alternative iteration by calculating some scalars
            // and by multiplying the reduced D by the second derivative matrix of
            // Q, where S holds the reduced D in the call of GGMULT.

        }
        case 100: {
            if (nact >= n - 1) {
                state = 190; break;
            }
            dredsq = ZERO;
            dredg = ZERO;
            gredsq = ZERO;
            for (int i = 0; i < n; i++) {
                if (xbdi[i] == ZERO) {
                    // Computing 2nd power
                    double d__1 = d__[i];
                    dredsq += d__1 * d__1;
                    dredg += d__[i] * gnew[i];
                    // Computing 2nd power
                    d__1 = gnew[i];
                    gredsq += d__1 * d__1;
                    s[i] = d__[i];
                } else {
                    s[i] = ZERO;
                }
            }
            itcsav = iterc;
            state = 210; break;
            // Let the search direction S be a linear combination of the reduced D
            // and the reduced G that is orthogonal to the reduced D.
        }
        case 120: {
            ++iterc;
            double temp = gredsq * dredsq - dredg * dredg;
            if (temp <= qred * 1e-4 * qred) {
                state = 190; break;
            }
            temp = Math.sqrt(temp);
            for (int i = 0; i < n; i++) {
                if (xbdi[i] == ZERO) {
                    s[i] = (dredg * d__[i] - dredsq * gnew[i]) / temp;
                } else {
                    s[i] = ZERO;
                }
            }
            sredg = -temp;

            // By considering the simple bounds on the variables, calculate an upper
            // bound on the tangent of half the angle of the alternative iteration,
            // namely ANGBD, except that, if already a free variable has reached a
            // bound, there is a branch back to label 100 after fixing that variable.

            angbd = ONE;
            iact = -1;
            for (int i = 0; i < n; i++) {
                if (xbdi[i] == ZERO) {
                    double tempa = xopt[i] + d__[i] - sl[i];
                    double tempb = su[i] - xopt[i] - d__[i];
                    if (tempa <= ZERO) {
                        ++nact;
                        xbdi[i] = ONEMIN;
                        state = 100; break L200;
                    } else if (tempb <= ZERO) {
                        ++nact;
                        xbdi[i] = ONE;
                        state = 100; break L200;
                    }
                    // Computing 2nd power
                    double d__1 = d__[i];
                    // Computing 2nd power
                    double d__2 = s[i];
                    double ssq = d__1 * d__1 + d__2 * d__2;
                    // Computing 2nd power
                    d__1 = xopt[i] - sl[i];
                    temp = ssq - d__1 * d__1;
                    if (temp > ZERO) {
                        temp = Math.sqrt(temp) - s[i];
                        if (angbd * temp > tempa) {
                            angbd = tempa / temp;
                            iact = i;
                            xsav = ONEMIN;
                        }
                    }
                    // Computing 2nd power
                    d__1 = su[i] - xopt[i];
                    temp = ssq - d__1 * d__1;
                    if (temp > ZERO) {
                        temp = Math.sqrt(temp) + s[i];
                        if (angbd * temp > tempb) {
                            angbd = tempb / temp;
                            iact = i;
                            xsav = ONE;
                        }
                    }
                }
            }

            // Calculate HHD and some curvatures for the alternative iteration.

            state = 210; break;
        }
        case 150: {
            double shs = ZERO;
            double dhs = ZERO;
            double dhd = ZERO;
            for (int i = 0; i < n; i++) {
                if (xbdi[i] == ZERO) {
                    shs += s[i] * hs[i];
                    dhs += d__[i] * hs[i];
                    dhd += d__[i] * hred[i];
                }
            }

            // Seek the greatest reduction in Q for a range of equally spaced values
            // of ANGT in [0,ANGBD], where ANGT is the tangent of half the angle of
            // the alternative iteration.

            double redmax = ZERO;
            double isav = 0;
            double redsav = ZERO;
            int iu = (int) (angbd * 17. + 3.1);
            for (int i = 1; i <= iu; i++) {
                angt = angbd * (double) i / (double) iu;
                double sth = (angt + angt) / (ONE + angt * angt);
                double temp = shs + angt * (angt * dhd - dhs - dhs);
                double rednew = sth * (angt * dredg - sredg - HALF * sth * temp);
                if (rednew > redmax) {
                    redmax = rednew;
                    isav = i;
                    rdprev = redsav;
                } else if (i == isav + 1) {
                    rdnext = rednew;
                }
                redsav = rednew;
            }

            // Return if the reduction is zero. Otherwise, set the sine and cosine
            // of the angle of the alternative iteration, and calculate SDEC.

            if (isav == 0) {
                state = 190; break;
            }
            if (isav < iu) {
                double temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext);
                angt = angbd * ((double) isav + HALF * temp) / (double) iu;
            }
            double cth = (ONE - angt * angt) / (ONE + angt * angt);
            double sth = (angt + angt) / (ONE + angt * angt);
            double temp = shs + angt * (angt * dhd - dhs - dhs);
            double sdec = sth * (angt * dredg - sredg - HALF * sth * temp);
            if (sdec <= ZERO) {
                state = 190; break;
            }

            // Update GNEW, D and HRED. If the angle of the alternative iteration
            // is restricted by a bound on a free variable, that variable is fixed
            // at the bound.

            dredg = ZERO;
            gredsq = ZERO;
            for (int i = 0; i < n; i++) {
                gnew[i] = gnew[i] + (cth - ONE) * hred[i] + sth * hs[i];
                //gnew[i] += (cth - ONE) * hred[i] + sth * hs[i]; //???
                if (xbdi[i] == ZERO) {
                    d__[i] = cth * d__[i] + sth * s[i];
                    dredg += d__[i] * gnew[i];
                    // Computing 2nd power
                    double d__1 = gnew[i];
                    gredsq += d__1 * d__1;
                }
                hred[i] = cth * hred[i] + sth * hs[i];
            }
            qred += sdec;
            if (iact >= 0 && isav == iu) {
                ++nact;
                xbdi[iact] = xsav;
                state = 100; break;
            }

            // If SDEC is sufficiently small, then RETURN after setting XNEW to
            // XOPT+D, giving careful attention to the bounds.

            if (sdec > qred * .01) {
                state = 120; break;
            }
        }
        case 190: {
            dsq.value = ZERO;
            for (int i = 0; i < n; i++) {
                // Computing MAX
                // Computing MIN
                double d__3 = xopt[i] + d__[i];
                double d__4 = su[i];
                double d__1 = Math.min(d__3,d__4);
                double d__2 = sl[i];
                xnew[i] = Math.max(d__1,d__2);
                if (xbdi[i] == ONEMIN) {
                    xnew[i] = sl[i];
                }
                if (xbdi[i] == ONE) {
                    xnew[i] = su[i];
                }
                d__[i] = xnew[i] - xopt[i];
                // Computing 2nd power
                d__1 = d__[i];
                dsq.value += d__1 * d__1;
            }
            return;
            // The following instructions multiply the current S-vector by the second
            // derivative matrix of the quadratic model, putting the product in HS.
            // They are reached from three different parts of the software above and
            // they can be regarded as an external subroutine.
        }
        case 210: {
            int ih = 0;
            for (int j = 0; j < n; j++) {
                hs[j] = ZERO;
                for (int i = 0; i <= j; i++) {
                    if (i < j) {
                        hs[j] += hq[ih] * s[i];
                    }
                    hs[i] += hq[ih] * s[j];
                    ih++;
                }
            }
            for (int k = 0; k < npt; k++) {
                if (pq[k] != ZERO) {
                    double temp = ZERO;
                    for (int j = 0; j < n; j++) {
                        temp += xpt[k + j * npt] * s[j];
                    }
                    temp *= pq[k];
                    for (int i = 0; i < n; i++) {
                        hs[i] += temp * xpt[k + i * npt];
                    }
                }
            }
            if (crvmin.value != ZERO) {
                state = 50; break;
            }
            if (iterc > itcsav) {
                state = 150; break;
            }
            for (int i = 0; i < n; i++) {
                hred[i] = hs[i];
            }
            state = 120; break;
        }}
    } // trsbox

    // ----------------------------------------------------------------------------------------

    /**
     *     The arrays BMAT and ZMAT are updated, as required by the new position
     *     of the interpolation point that has the index KNEW. The vector VLAG has
     *     N+NPT components, set on entry to the first NPT and last N components
     *     of the product Hw in equation (4.11) of the Powell (2006) paper on
     *     NEWUOA. Further, BETA is set on entry to the value of the parameter
     *     with that name, and DENOM is set to the denominator of the updating
     *     formula. Elements of ZMAT may be treated as ZERO if their moduli are
     *     at most ZTEST. The first NDIM elements of W are used for working space.
     * @param beta
     * @param denom
     * @param knew
     */
    private void update(
            double beta,
            double denom,
            int knew
    ) {
        double ztest = ZERO;
        for (int k = 0; k < npt; k++) {
            for (int j = 0; j < nptm; j++) {
                // Computing MAX
                double d__2 = ztest;
                double d__1 = zmat[k + j * npt];
                double d__3 = Math.abs(d__1);
                ztest = Math.max(d__2,d__3);
            }
        }
        ztest *= 1e-20;

        // Apply the rotations that put zeros in the KNEW-th row of ZMAT.

        for (int j = 1; j < nptm; j++) {
            double d__1 = zmat[knew + j * npt];
            if (Math.abs(d__1) > ztest) {
                // Computing 2nd power
                d__1 = zmat[knew];
                // Computing 2nd power
                double d__2 = zmat[knew + j * npt];
                double temp = Math.sqrt(d__1 * d__1 + d__2 * d__2);
                double tempa = zmat[knew] / temp;
                double tempb = zmat[knew + j * npt] / temp;
                for (int i = 0; i < npt; i++) {
                    temp = tempa * zmat[i] + tempb * zmat[i + j *
                            npt];
                    zmat[i + j * npt] = tempa * zmat[i + j * npt] -
                            tempb * zmat[i];
                    zmat[i] = temp;
                }
            }
            zmat[knew + j * npt] = ZERO;
        }

        // Put the first NPT components of the KNEW-th column of HLAG into W,
        // and calculate the parameters of the updating formula.

        for (int i = 0; i < npt; i++) {
            w[i] = zmat[knew] * zmat[i];
        }
        double alpha = w[knew];
        double tau = vlag[knew];
        vlag[knew] -= ONE;

        // Complete the updating of ZMAT.

        double temp = Math.sqrt(denom);
        double tempb = zmat[knew] / temp;
        double tempa = tau / temp;
        for (int i = 0; i < npt; i++) {
            zmat[i] = tempa * zmat[i] -
                    tempb * vlag[i];
        }

        // Finally, update the matrix BMAT.

        for (int j = 0; j < n; j++) {
            int jp = npt + j;
            w[jp] = bmat[knew + j * ndim];
            tempa = (alpha * vlag[jp] - tau * w[jp]) / denom;
            tempb = (-beta * w[jp] - tau * vlag[jp]) / denom;
            for (int i = 0; i <= jp; i++) {
                bmat[i + j * ndim] = bmat[i + j * ndim] + tempa *
                        vlag[i] + tempb * w[i];
                //bmat[i + j * ndim] += tempa *
                //		vlag[i] + tempb * w[i]; ///???
                if (i >= npt) {
                    bmat[jp + (i - npt) * ndim] =  bmat[i + j * ndim];
                }
            }
        }
    } // update

    /**
     * Checks dimensions and values of boundaries and inputSigma if defined.
     */
    private void checkParameters() {
        // Check problem dimension.
        if (n < 2) {
            throw new NumberIsTooSmallException(n, 2, true);
        }
        // Check number of interpolation points.
        final int[] nPointsInterval = { n + 2, (n + 2) * (n + 1) / 2 };
        if (npt < nPointsInterval[0] ||
                npt > nPointsInterval[1]) {
            throw new OutOfRangeException(LocalizedFormats.NUMBER_OF_INTERPOLATION_POINTS,
                    npt,
                    nPointsInterval[0],
                    nPointsInterval[1]);
        }
        // Initialize bound differences.
        boundDifference = new double[n];
    }

    // auxiliary subclasses

    /**
     * Double reference
     */
    private static class DoubleRef {
        /**
         * stored double value.
         */
        private double value;

        /**
         * @param value stored double value.
         */
        DoubleRef(double value) {
            this.value = value;
        }
    }

    /**
     * Integer reference
     */
    private static class IntRef {
        /**
         * stored int value.
         */
        private int value;

        /**
         * @param value stored int value.
         */
        IntRef(int value) {
            this.value = value;
        }
    }

    /**
     * @param n dimension.
     * @param value value set for each element.
     * @return array containing n values.
     */
    private static double[] point(int n, double value) {
        double[] ds = new double[n];
        Arrays.fill(ds, value);
        return ds;
    }

}
