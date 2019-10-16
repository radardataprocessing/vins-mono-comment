#include "marginalization_factor.h"

/*
 * cost function depends on parameter block
 * loss function is a scalar function that is used to reduce the influence of outliers on the solution of the optimazation problem
 * use the cost function to compute the residual and jacobian
 * if loss_function is not null, use the loss function to weight jacobian and residual
 */
void ResidualBlockInfo::Evaluate()
{
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    /*
     * For least squares problems where the minimization may encounter input terms that contain outliers, that is, completely bogus measurements, it
     * is important to use a loss function that reduces their influence.
     * Using a robust loss function, the cost for large residuals is reduced. Outlier terms get down-weighted so they do not overly influence the final solution
     * The key method is void LossFunction::Evaluate(double s, double out[3]), which given a non-negative scalar s, computes
     * out = [rho(s), rho(s).first_order_derivative(), rho(s).second_order_derivative() ]
     * Here the convention is that the contribution of a term to the cost function is given by 1/2*rho(s), where s = ||fi||.square(). Calling the method
     * with a negative value of s is an error and the implementations are not required to handle that case.
     * Most sane choices of rho satisfy 4 conditions:
     * (1) rho(0) = 0
     * (2) rho(0).first_order_derivative() = 0
     * (3) rho(s).first_order_derivative() < 1 in the outlier region
     * (4) rho(s).second_order_derivative() < 0 in the outlier region
     */
    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    // std::unordered_map<long, double *> parameter_block_data;
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        
        delete factors[i]->cost_function;

        delete factors[i];
    }
}

/*
 * 1. emplace_back the residual block info to the vector<ResidualBlockInfo *> factors
 * 2. use the address of every parameter_block and its corresponding size to construct a pair to put 
 *    into the unordered_map parameter_block_size
 * 3. use the address of every drop_set and 0 to construct a pair to put into the unordered_map
 *    parameter_block_idx
 */
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    // std::vector<ResidualBlockInfo *> factors;
    factors.emplace_back(residual_block_info);

    // std::vector<double *> parameter_blocks;    ceres::CostFunction *cost_function;
    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        // std::unordered_map<long, int> parameter_block_size; //global size
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }
    // I think drop_set means parameters to be dropped 
    // set the parameter block index of the drop set to be 0
    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        // std::vector<int> drop_set;      std::unordered_map<long, int> parameter_block_idx; //local size
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
    // until now, parameter_block_idx just contain the drop_set, parameter_block_size contains all the parameters
}

/*
 * put the data in parameter_blocks of every ResidualBlockInfo into parameter_block_data
 */
void MarginalizationInfo::preMarginalize()
{
    // std::vector<ResidualBlockInfo *> factors;
    for (auto it : factors)
    {
        it->Evaluate();

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            // std::unordered_map<long, double *> parameter_block_data;
            // std::vector<double *> parameter_blocks;
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                // memcpy copy from the second param to the first param
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

/*
 * struct ThreadsStruct
 * {
 *     std::vector<ResidualBlockInfo *> sub_factors;
 *     Eigen::MatrixXd A;
 *     Eigen::VectorXd b;
 *     std::unordered_map<long, int> parameter_block_size; //global size
 *     std::unordered_map<long, int> parameter_block_idx; //local size
 * };
 */
/*
 * 
 */
void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);
    for (auto it : p->sub_factors)// for every element it in std::vector<ResidualBlockInfo *> sub_factors
    {
        // it is a ResidualBlockInfo* object
        // parameter_blocks is a std::vector<double*> object
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)// for every parameter_block corresponding to the residual block it
        {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];// local size  index
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];// global size   size
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                //get the jacobian of block i and block j
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

/*
 * for the following equation
 *  |       pa        pb | | delta_xa |   | ga |
 *  | pb.transpose()  pc |*| delta_xb | = | gb |      -------------------(1)
 * if we want to marginalize delta_xa in (1)
 * |             I                 0 | |       pa        pb | | delta_xa |   |             I                 0 | | ga |
 * | -pb.transpose()*pa.inverse()  I |*| pb.transpose()  pc |*| delta_xb | = | -pb.transpose()*pa.inverse()  I |*| gb | ---------------(2)
 * which can be written as
 * | pa                 pb                 | | delta_xa |   |                 ga                |
 * | 0   pc-pb.transpose()*pa.inverse()*pb |*| delta_xb | = | gb-pb.transpose()*pa.inverse()*ga |   ---------------------(3)
 * observe the second row of the (3) equation, we can get
 * (pc-pb.transpose()*pa.inverse()*pb)*delta_xb=gb-pb.transpose()*pa.inverse()*ga
 * here, we can see that delta_xa has been marginalized
 */
void MarginalizationInfo::marginalize()
{
    int pos = 0;
    for (auto &it : parameter_block_idx)//std::unordered_map<long, int> parameter_block_idx; //local size
    {
        it.second = pos;
        // localSize is a function, if its input is 7, return 6; if its input is other number, return the input
        pos += localSize(parameter_block_size[it.first]);
    }

    // the parameters to be marginalized has smaller index, 
    m = pos;// the num of params tobe marg

    // from here, we can infer that parameter_block_size has more elements than parameter_block_idx
    // then use the size in parameter_block_size to compute index and put the index into parameter_block_idx
    for (const auto &it : parameter_block_size)// if parameter_block_size has addr that parameter_block_idx does not have
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())// if we can not find this it in parameter_block_idx
        {
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }
    // the elements that parameter_block_size have but parameter_block_idx does not have means the elements tobe kept

    // the parameters to be kept has bigger index than the parameters tobe marginalized
    n = pos - m;// the num of params tobe kept

    //ROS_DEBUG("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);// pos means the sum of sizes of all params
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread


    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
    /*
     * devide the factors into NUM_THREADS parts and push them into sub_factors of the threadsstruct, so that we can solve A
     * and b in multi threads
     */
    for (auto it : factors)// devide the factors into NUM_THREADS parts and push them into sub_factors of the threadsstruct
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    // Constructs threads using the array of ThreadStruct whose size is NUM_THREADS
    // every thread will use its subfactors to fill some blocks of matrix A and b, then we can sum them up to get the whole matrix
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        // in this case, do we just compute the cross-covariance of parameters that influence the same ResidualBlockInfo
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    for( int i = NUM_THREADS - 1; i >= 0; i--)  
    {
        pthread_join( tids[i], NULL ); 
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());

    /*
     * SelfAdjointEigenSolver computes eigenvalues and eigenvectors of selfadjoint matrices
     * A is selfadjoint if it equals its adjoint. For real matrices, this means that the matrix is symmetric: it equals its transpose. This class
     * computes the eigenvalues and eigenvectors of a selfadjoint matrix. These are the scalars lambda and vectors v such that A*v=lambda*v. The
     * eigenvalues of a selfadjoint matrix are always real. If D is a diagonal matrix with the eigenvalues on the diagonal, and V is a matrix with 
     * the eigenvectors as its columns, then A=VDV.inverse()(for selfadjoint matrices, the matrix V is always invertible). This is called eigendecomposition
     * 
     * if A = V*D*V.inverse()   B = V*D.inverse()*V.inverse()
     * A*B = V*D*V.inverse()*V*D.inverse()*V.inverse()
     *     = V*D*(V.inverse()*V)*D.inverse()*V.inverse()
     *     = V*D*D.inverse()*V.inverse()
     *     = V*(D*D.inverse())*V.inverse()
     *     = V*V.inverse()
     *     = I
     * B*A = I
     * B = A.inverse()
     */
    //TODO
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    // array() returns an Eigen::ArrayBase Array expression of this matrix
    // Eigen Array provides some interface to operate on each element independently
    // (R.array() < s).select(P,Q) ==> (R < s ? P : Q) also is a operator on each element not the whole matrix
    // for every eigenvalues decomposed from the matrix saes, if the eigenvalue is bigger than eps, put the inverse of the eigen 
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    /* 
     * currently, the matrix A is an n*n matrix, S is a vector whose size is the same as the number of eigen values, if an eigen value is bigger than 
     * eps, put it into S; else put 0 instead into S. S_inv is a vector whose size is the same as the number of eigen values, if an eigen value is 
     * bigger than eps, put its inverse into S_inv; else put 0 instead into S
     */
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    // cwise means to operate on each element independently
    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    // question :: is linearized_jacobians the LLT decomposion of matrix A
    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

/*
 * put the infomation of the parameter blocks to be kept into the corresponding vectors, compute the sum of all the kept parameter
 * blocks and at last return keep_block_addr
 */
std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        if (it.second >= m)
        {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]); // std::unordered_map<long, double *> parameter_block_data;
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    /*
     * template< class InputIt, class T >
     * Ｔ accumulate( InputIt first, InputIt last, T init);
     * Computes the sum of the given value init and the elements in range [first, last)
     */
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}


/*
 * namespace ceres {
 * 
 * // This class implements the computation of the cost (a.k.a. residual) terms as
 * // a function of the input (control) variables, and is the interface for users
 * // to describe their least squares problem to Ceres. In other words, this is the
 * // modelling layer between users and the Ceres optimizer. The signature of the
 * // function (number and sizes of input parameter blocks and number of outputs)
 * // is stored in parameter_block_sizes_ and num_residuals_ respectively. User
 * // code inheriting from this class is expected to set these two members with the
 * // corresponding accessors. This information will be verified by the Problem
 * // when added with AddResidualBlock().
 * class CERES_EXPORT CostFunction {
 *  public:
 *   CostFunction() : num_residuals_(0) {}
 * 
 *   virtual ~CostFunction() {}
 * 
 *   // Inputs:
 *   //
 *   // parameters is an array of pointers to arrays containing the
 *   // various parameter blocks. parameters has the same number of
 *   // elements as parameter_block_sizes_.  Parameter blocks are in the
 *   // same order as parameter_block_sizes_.i.e.,
 *   //
 *   //   parameters_[i] = double[parameter_block_sizes_[i]]
 *   //
 *   // Outputs:
 *   //
 *   // residuals is an array of size num_residuals_.
 *   //
 *   // jacobians is an array of size parameter_block_sizes_ containing
 *   // pointers to storage for jacobian blocks corresponding to each
 *   // parameter block. Jacobian blocks are in the same order as
 *   // parameter_block_sizes, i.e. jacobians[i], is an
 *   // array that contains num_residuals_* parameter_block_sizes_[i]
 *   // elements. Each jacobian block is stored in row-major order, i.e.,
 *   //
 *   //   jacobians[i][r*parameter_block_size_[i] + c] =
 *   //                              d residual[r] / d parameters[i][c]
 *   //
 *   // If jacobians is NULL, then no derivatives are returned; this is
 *   // the case when computing cost only. If jacobians[i] is NULL, then
 *   // the jacobian block corresponding to the i'th parameter block must
 *   // not to be returned.
 *   //
 *   // The return value indicates whether the computation of the
 *   // residuals and/or jacobians was successful or not.
 *   //
 *   // This can be used to communicate numerical failures in jacobian
 *   // computations for instance.
 *   //
 *   // A more interesting and common use is to impose constraints on the
 *   // parameters. If the initial values of the parameter blocks satisfy
 *   // the constraints, then returning false whenever the constraints
 *   // are not satisfied will prevent the solver from moving into the
 *   // infeasible region. This is not a very sophisticated mechanism for
 *   // enforcing constraints, but is often good enough.
 *   //
 *   // Note that it is important that the initial values of the
 *   // parameter block must be feasible, otherwise the solver will
 *   // declare a numerical problem at iteration 0.
 *   virtual bool Evaluate(double const* const* parameters,
 *                         double* residuals,
 *                         double** jacobians) const = 0;
 * 
 *   const std::vector<int32>& parameter_block_sizes() const {
 *     return parameter_block_sizes_;
 *   }
 * 
 *   int num_residuals() const {
 *     return num_residuals_;
 *   }
 * 
 *  protected:
 *   std::vector<int32>* mutable_parameter_block_sizes() {
 *     return &parameter_block_sizes_;
 *   }
 * 
 *   void set_num_residuals(int num_residuals) {
 *     num_residuals_ = num_residuals;
 *   }
 * 
 *  private:
 *   // Cost function signature metadata: number of inputs & their sizes,
 *   // number of outputs (residuals).
 *   std::vector<int32> parameter_block_sizes_;
 *   int num_residuals_;
 *   CERES_DISALLOW_COPY_AND_ASSIGN(CostFunction);
 * };
 * 
 * }  // namespace ceres
 */
/*
 * push the sizes of keep blocks into parameter_block_sizes_, and compute the sum of all the sizes, then set num_residuals_ 
 * tobe n of the marginalization_info, here n means size of the parameters tobe kept
 */
MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);
};

/*
 * 
 */
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    int n = marginalization_info->n; // the size of parameters to be kept
    int m = marginalization_info->m; // the size of parameters to be marginalized
    Eigen::VectorXd dx(n);
    // std::vector<int> keep_block_size; global size      std::vector<int> keep_block_idx; local size     std::vector<double *> keep_block_data;
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];
        int idx = marginalization_info->keep_block_idx[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else // if this parameter block represents pose, use quaternion to process it
        {
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            // .vec() means the imaginary part of the quaternion
            // Utility::positify just returns the input quaternion
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians) // std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);// [idx+1, idx+local_size] columns
            }
        }
    }
    return true;
}
