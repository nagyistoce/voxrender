/* ===========================================================================

	Project: VoxRender - Matrix

	Defines a class for handling matrix operations

    Copyright (C) 2012 Lucas Sherman

	Lucas Sherman, email: LucasASherman@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

=========================================================================== */

// Begin definition
#ifndef VOX_MATRIX_H
#define VOX_MATRIX_H

// Max repeat macro depth
#ifndef VOX_MATRIX_LIMIT
#define VOX_MATRIX_LIMIT 10
#endif // VOX_MATRIX_LIMIT

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Geometry/Vector.h"
#include "VoxLib/Core/Preprocessor.h"

// API namespace
namespace vox
{
	/** CUDA Capable Generic Matrix Type */
	template<typename T, size_t M, size_t N> 
	struct Matrix
	{
		VOX_HOST_DEVICE Matrix( ) { }

        // Standard library container stuff
        typedef Vector<T,M>         value_type;
        typedef Vector<T,M> &       reference;
        typedef Vector<T,M> const&  const_reference;
        typedef Vector<T,M> *       pointer;
        typedef Vector<T,M> const*  const_pointer;
        typedef size_t              size_type;
        typedef size_t              difference_type;

        // Standard library iterator support
        typedef pointer         iterator;
        typedef const_pointer   const_iterator;

        typedef std::reverse_iterator<iterator>       reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

        VOX_HOST_DEVICE reference       front()       { return row[0]; }
        VOX_HOST_DEVICE const_reference front() const { return row[0]; }
   
        VOX_HOST_DEVICE reference       back()       { return row[N-1]; }
        VOX_HOST_DEVICE const_reference back() const { return row[N-1]; }

        VOX_HOST_DEVICE iterator        begin()       { return row; }
        VOX_HOST_DEVICE const_iterator  begin() const { return row; }
        
        VOX_HOST_DEVICE iterator       end()       { return row+N; }
        VOX_HOST_DEVICE const_iterator end() const { return row+N; }

        VOX_HOST_DEVICE reverse_iterator       rbegin()       { return reverse_iterator(end()); }
        VOX_HOST_DEVICE const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }

        VOX_HOST_DEVICE reverse_iterator       rend()       { return reverse_iterator(begin()); }
        VOX_HOST_DEVICE const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

        VOX_HOST_DEVICE static size_t size()  { return N; }
        VOX_HOST_DEVICE static bool   empty() { return false; }

        /** Returns a const reference to an identity matrix */
        VOX_HOST static inline Matrix const& identity()
        {
            static bool init = false;
            static Matrix result;

            if (!init)
            {
                init = true;

                // Construct the identity matrix
                for (size_t r = 0; r < N; r++)
                for (size_t c = 0; c < M; c++)
                {
                    if (c == r)
                    {
                        result.row[r][c] = T(1);
                    }
                    else
                    {
                        result.row[r][c] = T(0);
                    }
                }
            }

            return result;
        }

		/** Constructor */
		VOX_HOST_DEVICE explicit Matrix(const Vector<T,M>(&data)[N]) 
            { for (int i = 0; i < N; i++) row[i] = data[i]; }
        
// List form constructor templates
#define VOX_MATRIX_CONSTRUCTOR(z, n, _)             \
    Matrix( BOOST_PP_ENUM_PARAMS_Z(z, n,            \
        Vector<T BOOST_PP_COMMA() M> const& a) )    \
    {                                               \
        assign( BOOST_PP_ENUM_PARAMS_Z(z, n, a) );  \
    }

    //BOOST_PP_REPEAT_FROM_TO(2, VOX_MATRIX_LIMIT, VOX_MATRIX_CONSTRUCTOR, _)

#undef VOX_MATRIX_CONSTRUCTOR
        
// List form assignment templates
#define VOX_MATRIX_ASSIGN(z, n, _)                  \
    void assign( BOOST_PP_ENUM_PARAMS_Z(z, n,       \
        Vector<T BOOST_PP_COMMA() M> const& a) )    \
    {                                               \
        PP_ARRAY_ASSIGN(z, n, (row) (a))            \
    }

    //BOOST_PP_REPEAT_FROM_TO(2, VOX_MATRIX_LIMIT, VOX_MATRIX_ASSIGN, _)

#undef VOX_MATRIX_ASSIGN

		/** Matrix - Vector multiplication */
		VOX_HOST_DEVICE inline Vector<T,M> operator*(Vector<T,M> const& vec) const
		{
			Vector<T,M> result;
			for (int i = 0; i < N; i++)
				result[i] = row[i] * vec;
			return result;
		}
        
		/** Matrix - Equality comparison operator */
		VOX_HOST_DEVICE inline bool operator==(Matrix const& rhs) const
		{
			for (int i = 0; i < N; i++)
				if (row[i] != rhs.row[i]) 
					return false;
			return true;
		}
        
		/** Matrix - Not-Equal comparison operator */
		VOX_HOST_DEVICE inline bool operator!=(Matrix const& rhs) const
		{
			for (int i = 0; i < N; i++)
				if (row[i] != rhs.row[i]) 
					return true;
			return true;
		}

		/** Matrix - array style access operator */
		VOX_HOST_DEVICE inline const_reference operator[]( int i ) const
		{
			return row[i];
		}
        
		/** Matrix - array style access operator */
		VOX_HOST_DEVICE inline Vector<T,M>& operator[]( int i )
		{
			return row[i];
		}

		Vector<T,M> row[N]; ///< Matrix rows
	};

#undef VOX_MATRIX_LIMIT

	/** Vector stream operator **/
	template< typename T, int M, int N >
    VOX_HOST std::ostream &operator<<( std::ostream &os, const Matrix<T,M,N> &mat ) 
	{
		os << "["; 
		for( int i = 0; i < N-1; i++ )
			os << mat[i] << ","; 
        os << mat[N-1] << "]";
		return os;
	}

	// Standard matrix types
	typedef Matrix<int,4,4>    Matrix2x2;
	typedef Matrix<int,3,3>    Matrix3x3;
	typedef Matrix<int,4,4>    Matrix4x4;
	typedef Matrix<float,4,4>  Matrix2x2f;
	typedef Matrix<float,3,3>  Matrix3x3f;
	typedef Matrix<float,4,4>  Matrix4x4f;
	typedef Matrix<size_t,4,4> Matrix2x2u;
	typedef Matrix<size_t,3,3> Matrix3x3u;
	typedef Matrix<size_t,4,4> Matrix4x4u;
}

// End Definition
#endif // VOX_MATRIX_H