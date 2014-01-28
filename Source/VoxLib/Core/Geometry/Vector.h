/* ===========================================================================

	Project: VoxRender - Vector

	Defines a class for handling traditional mathematical vector operations.

    Copyright (C) 2012-2013 Lucas Sherman

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
#ifndef VOX_VECTOR_H
#define VOX_VECTOR_H

// Max repeat macro depth
#ifndef VOX_VECTOR_LIMIT
#define VOX_VECTOR_LIMIT 10
#endif // VOX_VECTOR_LIMIT

// Matrix prototype
template< typename T, size_t M, size_t N > struct Matrix;

// Include Dependencies
#include "VoxLib/Core/CudaCommon.h"
#include "VoxLib/Core/Preprocessor.h"
#include "VoxLib/Core/Types.h"

// API namespace
namespace vox
{
	/** 
     * CUDA Capable Generic Vector Type 
     *
     * This class provides a device independent representational format for vector data. The intent of the class is
     * not to provide a high effeciency interface for vector operations, but to provide a framework for specifying 
     * data structures and operators independent of the device (GPU or CPU).
     */
    template<typename T, size_t N> struct Vector
	{
        VOX_HOST_DEVICE Vector() { } 

        // Standard library container stuff
        typedef T           value_type;
        typedef T &         reference;
        typedef T const&    const_reference;
        typedef T *         pointer;
        typedef T const*    const_pointer;
        typedef size_t      size_type;
        typedef size_t      difference_type;

        // Standard library iterator support
        typedef pointer         iterator;
        typedef const_pointer   const_iterator;

        typedef std::reverse_iterator<iterator>       reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

        VOX_HOST_DEVICE reference       front()       { return coord[0]; }
        VOX_HOST_DEVICE const_reference front() const { return coord[0]; }
   
        VOX_HOST_DEVICE reference       back()       { return coord[N-1]; }
        VOX_HOST_DEVICE const_reference back() const { return coord[N-1]; }

        VOX_HOST_DEVICE iterator       begin()       { return coord; }
        VOX_HOST_DEVICE const_iterator begin() const { return coord; }
        
        VOX_HOST_DEVICE iterator       end()       { return coord+N; }
        VOX_HOST_DEVICE const_iterator end() const { return coord+N; }

        VOX_HOST_DEVICE reverse_iterator       rbegin()       { return reverse_iterator(end()); }
        VOX_HOST_DEVICE const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }

        VOX_HOST_DEVICE reverse_iterator       rend()       { return reverse_iterator(begin()); }
        VOX_HOST_DEVICE const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

        VOX_HOST_DEVICE static size_type size()  { return N; }
        VOX_HOST_DEVICE static bool      empty() { return false; }

        /** Constructs a vector and initializes each entry to the given value */
		VOX_HOST_DEVICE explicit Vector(T const& value) { fill(value); }

        /** Constructs a vector whose entries match the values in the input array */
		VOX_HOST_DEVICE explicit Vector(const T(&data)[N]) 
        { 
            for (size_t i = 0; i < N; i++) coord[i] = data[i]; 
        }

// List form constructor templates
#define VOX_VECTOR_CONSTRUCTOR(z, n, _)                     \
    VOX_HOST_DEVICE                                         \
    Vector( BOOST_PP_ENUM_PARAMS_Z(z, n, T const& a) )      \
    {                                                       \
        assign( BOOST_PP_ENUM_PARAMS_Z(z, n, a) );          \
    }                                                       

    BOOST_PP_REPEAT_FROM_TO(2, VOX_VECTOR_LIMIT, VOX_VECTOR_CONSTRUCTOR, _)

#undef VOX_VECTOR_CONSTRUCTOR

        /** Constructs a type casted vector */
        template<typename S> explicit Vector(Vector<S,N> const& base)
        {
            for (size_t i = 0; i < N; i++)
            {
                coord[i] = static_cast<T>(base[i]);
            }
        }

        /**
         * Fold operator
         *
         * Performs a fold operation on the vector. A fold operation
         * takes a seed value and iterates through the vector elements
         * applying op to the running seed value and the current element.
         *
         * @param seed The seed value for the fold sequence
         * @param op The fold operator
         */
        template<typename R>                           
        R fold(R const& seed, R (*op)(R const&, T const&)) const                                                       
        {                                                           
            R result = seed;                                         
            for (size_t i = 0; i < N; i++) 
                result = op(result, coord[i]); 
            return result;                                           
        }

        /**
         * Fold operator
         *
         * Performs a fold operation on the vector. A fold operation
         * iterates through the vector elements applying op.
         *
         * @param seed The seed value for the fold sequence
         * @param op The fold operator
         */
        T fold(T (*op)(T const&, T const&)) const
        {
            T res = coord[0];
            for (size_t i = 1; i < N; i++)
                res = op(res, coord[i]);
            return res;
        }

        /**
         * Fold operator
         *
         * Performs a fold operation on the vector. A fold operation
         * takes a seed value and iterates through the vector elements
         * applying op to the running seed value and the current element.
         *
         * @param seed [in] [out] The seed value for the fold sequence
         * @param op The fold operator
         * @returns The return value of the final fold operation
         */
        template<typename R>
        R& fold(R& seed, R& (*op)(R&, T const&)) const
        {                                       
            for (size_t i = 0; i < N; i++)                                             
                seed = op(seed, coord[i]);
            return seed;
        }

        /**
         * Map operator by standard assignment
         *
         * Performs a map operation on the vector. A map operation
         * takes an operator and iterates through the vector elements
         * applying the operator to the each element in succession.
         *
         * @param op The mapping operator
         */
        VOX_HOST Vector& map(std::function<T(T const&)> op)
        {                                       
            for (size_t i = 0; i < N; i++) coord[i] = std::move(op(coord[i]));

            return *this;
        }

        /**
         * Map operator
         *
         * Essentially an overload of map avoiding function type inferences.
         *
         * @param op The mapping operator
         */
        VOX_HOST Vector& mapAssign(std::function<void(T&)> op)
        {                                       
            for (size_t i = 0; i < N; i++) op(coord[i]);

            return *this;
        }

        /**
         * Map operator
         *
         * Map operation which constructs a new vector for the result
         *
         * @param op The map operator
         */
        template<typename R>
        VOX_HOST Vector<R,N> mapCopy(std::function<R(T const&)> op) const
        {       
            Vector<R,N> result;                    
            for (size_t i = 0; i < N; i++)                                             
                result[i] = std::move(op(coord[i]));
            return result;
        }
        
        /**
         * Map operator
         *
         * Map operation which constructs a new vector for the result
         *
         * @param op The map operator
         */
        template<typename R>
        VOX_HOST Vector<R,N> mapCopyAssign(std::function<void(R &,T const&)> op) const
        {       
            Vector<R,N> result;                    
            for (size_t i = 0; i < N; i++) op(result[i], coord[i]);
            return result;
        }

// List form constructor templates
#define VOX_VECTOR_CONSTRUCTOR(z, n, _)                         \
    VOX_HOST_DEVICE                                             \
    void assign( BOOST_PP_ENUM_PARAMS_Z(z, n, T const& a) )     \
    {                                                           \
        PP_ARRAY_ASSIGN(z, n, (coord) (a))                      \
    }                                                       

    BOOST_PP_REPEAT_FROM_TO(2, VOX_VECTOR_LIMIT, VOX_VECTOR_CONSTRUCTOR, _)

#undef VOX_VECTOR_CONSTRUCTOR

        /**
         * Assigns the specified value to each element of the vector
         *
         * @param value The value to fill the vector with
         */
        VOX_HOST_DEVICE inline void fill(T const& value) 
        { 
            for (size_t i = 0; i < N; i++) coord[i] = value; 
        }

		/** Vector magnitude */
		VOX_HOST_DEVICE inline T length() const { return sqrt( lengthSquared( ) ); }

		/** Vector magnitude squared */
		VOX_HOST_DEVICE inline T lengthSquared() const
		{
			T dist = coord[0]*coord[0];
			for (size_t i = 1; i < N; i++)
				dist += coord[i]*coord[i];
			return dist;
		}

		/** Normalizes the vector and returns a reference to self */
		VOX_HOST_DEVICE inline Vector& normalize() { return *this /= length(); }

        /** Returns a normalized copy of the vector */
		VOX_HOST_DEVICE inline Vector normalized() const { return *this / length(); }

        /** Unary subtraction operator */
        VOX_HOST_DEVICE Vector operator-()
        {
            Vector<T,N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = - coord[i];
            }
            return result;
        }

		/** Assignment Operator **/
		template< int M >
		VOX_HOST_DEVICE inline const Vector<T,N>& operator=(Vector<T,M> rhs)
		{
			size_t max = M > N ? M : N;
			for (size_t i = 0; i < max; i++)
				coord[i] = rhs.coord[i];
			return *this;
		}

		/** Vector multiplication - { a1*b1, a2*b2, ... } etc */
		VOX_HOST_DEVICE inline Vector operator*(Vector rhs) const
		{
			Vector result;
			for (size_t i = 0; i < N; i++)
				result[i] = coord[i]*rhs[i];
			return result;
		}

        /** Vector multiplication - { a1*b1, a2*b2, ... } etc */
		VOX_HOST_DEVICE inline Vector operator*=(Vector rhs)
		{
			for (size_t i = 0; i < N; i++) 
                coord[i] *= rhs[i];
			return result;
		}

		/** Vector division - { a1/b1, a2/b2, ... } etc */
		VOX_HOST_DEVICE inline Vector operator/(Vector rhs) const
		{
			Vector result;
			for (size_t i = 0; i < N; i++)
				result[i] = coord[i]/rhs[i];
			return result;
		}

        /** Vector division - { a1/b1, a2/b2, ... } etc */
		VOX_HOST_DEVICE inline Vector operator/=(Vector rhs)
		{
			for (size_t i = 0; i < N; i++) 
                coord[i] /= rhs[i];
			return result;
		}

		/** Matrix multiplication */
		template<size_t M> Vector<T,M> operator* (Matrix<T,M,N> const& rhs) const;
		Vector& operator*=(Matrix<T,N,N> const& rhs);
		
		/** Pseudo resize operator for vector */ 
        template<size_t M> Vector<T,M> resize() const
		{
			Vector<T,M> result;
			size_t min = M < N ? M : N;
			for (size_t i = 0; i < min; i++)
				result[i] = coord[i];
			for (size_t i = min; i < M; i++)
				result[i] = 0;
			return result;
		}

		/** Type conversion operator */
		template<typename S> operator Vector<S,N>() 
		{
			Vector<S,N> result;
			for (size_t i = 0; i < N; i++)
				result[i] = S(coord[i]);
			return result;
		}
        
        /** Division Operator */
		VOX_HOST_DEVICE inline Vector operator/( float rhs ) const
		{
			Vector<T,N> result;
			T inv = static_cast<T>(1) / rhs;
			for (size_t i = 0; i < N; i++)
				result.coord[i] = coord[i] * inv;
			return result;
		}
        
        /** Division Operator */
		VOX_HOST_DEVICE inline Vector& operator/=(float rhs)
		{
			T inv = static_cast<T>(1) / rhs;
			for (size_t i = 0; i < N; i++)
				coord[i] *= inv;
			return *this;
		}
        
        /** Multiplication Operator */
		VOX_HOST_DEVICE inline Vector operator*(float rhs) const
		{
			Vector<T,N> result;
			for (size_t i = 0; i < N; i++)
				result.coord[i] = coord[i] * rhs;
			return result;
		}
        
        /** Multiplication Operator */
		VOX_HOST_DEVICE inline Vector& operator*=(float rhs)
		{
			for (size_t i = 0; i < N; i++)
				coord[i] *= rhs;
			return *this;
		}
        
        /** Addition Operator */
		VOX_HOST_DEVICE inline Vector operator+(const Vector& rhs) const
		{
			Vector<T,N> result;
			for (size_t i = 0; i < N; i++)
				result.coord[i] = coord[i] + rhs.coord[i];
			return result;
		}
        
        /** Addition Operator */
		VOX_HOST_DEVICE inline Vector& operator+=(const Vector& rhs)
		{
			for (size_t i = 0; i < N; i++)
				coord[i] += rhs[i];
			return *this;
		}
        
        /** Subtraction Operator */
		VOX_HOST_DEVICE inline Vector operator-(const Vector& rhs) const
		{
			Vector<T,N> result;
			for (size_t i = 0; i < N; i++)
				result.coord[i] = coord[i] - rhs.coord[i];
			return result;
		}

        /** Subtraction Operator */
		VOX_HOST_DEVICE inline Vector& operator-=(const Vector& rhs)
		{
			for (size_t i = 0; i < N; i++)
				coord[i] -= rhs[i];
			return *this;
		}
        
        /** Equals Operator */
		VOX_HOST_DEVICE inline bool operator==(const Vector& rhs) const
		{
			for (size_t i = 0; i < N; i++)
				if (coord[i] != rhs.coord[i]) 
					return false;
			return true;
		}
        
        /** Not Equal Operator */
		VOX_HOST_DEVICE inline bool operator!=(const Vector& rhs) const
		{
			for (size_t i = 0; i < N; i++)
				if (coord[i] != rhs.coord[i]) 
					return true;
			return true;
		}
        
        /** Less-than comparison operator */
        VOX_HOST_DEVICE bool operator<(Vector const& rhs) const
        {
            for (size_t i = 0; i < N; i++)
            {
                if (coord[i] < rhs.coord[i]) return true;
                else if (coord[i] > rhs.coord[i]) return false;
            }

            return false;
        }

        /** Less-than or equal-to comparison operator */
        VOX_HOST_DEVICE bool operator<=(Vector const& rhs) const
        {
            for (size_t i = 0; i < N; i++)
            {
                if (coord[i] < rhs.coord[i]) return true;
                else if (coord[i] > rhs.coord[i]) return false;
            }

            return true;
        }

        /** Greater-than comparison operator */
        VOX_HOST_DEVICE bool operator>(Vector const& rhs) const
        {
            for (size_t i = 0; i < N; i++)
            {
                if (coord[i] > rhs.coord[i]) return true;
                else if (coord[i] < rhs.coord[i]) return false;
            }

            return false;
        }

        /** Greater-than or equal-to comparison operator */
        VOX_HOST_DEVICE bool operator>=(Vector const& rhs) const
        {
            for (size_t i = 0; i < N; i++)
            {
                if (coord[i] > rhs.coord[i]) return true;
                else if (coord[i] < rhs.coord[i]) return false;
            }

            return true;
        }

        /** Array style access element access */
		VOX_HOST_DEVICE inline T const& operator[](size_t i) const { return coord[i]; }

        /** Array style access element access */
		VOX_HOST_DEVICE inline T& operator[](size_t i) { return coord[i]; }

	    /** 
         * Returns the distance between two position vectors.
         *
         * @param lhs [in] The first vector
         * @param rhs [in] The second vector 
         * @return The distance between lhs and rhs
         */
	    template<typename T, int N>
	    VOX_HOST_DEVICE inline Vector<T,N> distance( const Vector<T,N>& lhs, const Vector<T,N>& rhs ) 
		    { return (lhs - rhs).length( ); }

	    /** 
         * Returns the square of the distance between two position vectors.
         *
         * @param lhs [in] The first vector
         * @param rhs [in] The second vector 
         * @return The square of the distance between lhs and rhs
         */
	    template< typename T, int N >
	    VOX_HOST_DEVICE static inline T distanceSquared( const Vector<T,N>& lhs, const Vector<T,N>& rhs )  
		    { return (lhs - rhs).lengthSquared( ); }

	    /** 
         * Computes the cross product of two 3 dimensional vectors.
         * :TODO: Consider generalizing this to N while retaining 3D optimization
         *        Shouldn't even be defined here as it is now (Vector<T> static)
         *
         * @param lhs [in] The first vector of the cross product
         * @param rhs [in] The second vector of the cross product
         * @return The cross product of lhs and rhs (lhs X rhs)
         */
	    VOX_HOST_DEVICE static Vector<T,3> cross( 
            const Vector<T,3> &lhs, const Vector<T,3> &rhs ) 
	    {
		    Vector<T,3> result = 
                Vector<T,3>((lhs[1] * rhs[2]) - (lhs[2] * rhs[1]),
                            (lhs[2] * rhs[0]) - (lhs[0] * rhs[2]),
                            (lhs[0] * rhs[1]) - (lhs[1] * rhs[0])); 
		    return result;
	    }

	    /** 
         * Computes the dot product of two vectors.
         *
         * @param lhs [in] The first vector of the dot product
         * @param rhs [in] The second vector of the dot product
         * @return The dot product of lhs and rhs
         */
	    VOX_HOST_DEVICE static T
            dot(const Vector<T,N> &lhs, const Vector<T,N> &rhs) 
        {
			T dot = lhs.coord[0]*rhs.coord[0]; 
			for (size_t i = 1; i < N; i++)
				dot += lhs.coord[i]*rhs.coord[i];;
			return dot;
        }

        /** Alternative convention for dot product */
        VOX_HOST_DEVICE inline T dot(Vector const& lhs)
        {
            return Vector::dot(*this, lhs);
        }

        /** Returns a reference to the max element in the vector */
        VOX_HOST_DEVICE T & high()
        {
            T & result = coord[0];
            for (size_t i = 1; i < N; i++)
            {
                if (coord[i] > result) result = coord[i];
            }
            return result;
        }

        /** Returns a reference to the min element in the vector */
        VOX_HOST_DEVICE T & low()
        {
            T & result = coord[0];
            for (size_t i = 1; i < N; i++)
            {
                if (coord[i] < result) result = coord[i];
            }
            return result;
        }

        /** Const overload for maximum value function */
        VOX_HOST_DEVICE T const& high() const { return high(); }

        /** Const overload for minimum value function */
        VOX_HOST_DEVICE T const& low() const { return low(); }

		T coord[N]; ///< Vector coordinates
	};

#undef VOX_VECTOR_LIMIT

	/** Vector output stream operator */
	template< typename T, int N >
    VOX_HOST std::ostream& operator<<(std::ostream &os, const Vector<T,N> &vec) 
	{
		os << "["; 
		for( size_t i = 0; i < N-1; i++ )
        {
            os << vec.coord[i] << " ";
        } 
        os << vec.coord[N-1];
		os << "]";
		return os;
	}
 
	/** Vector input streaming operator */
	template<typename T, int N>
    VOX_HOST std::istream& operator>>(std::istream &is, Vector<T,N> &vec) 
	{
        bool delim = (is.peek() == '<' || is.peek() == '(' || is.peek() == '['); 
            
        if (delim) is.ignore(1);

		BOOST_FOREACH(auto & elem, vec.coord)
        {
            is >> elem;
            is.ignore(1); // :TODO: Ignore ws or ','
        } 
        
        if (delim) is.ignore(1);
		
        return is;
	}

	// Common vector types
	typedef Vector<int,2> Vector2; 
	typedef Vector<int,3> Vector3; 
	typedef Vector<int,4> Vector4;
	typedef Vector<float,2> Vector2f;
	typedef Vector<float,3> Vector3f;
	typedef Vector<float,4> Vector4f;
	typedef Vector<unsigned int,2> Vector2u;
	typedef Vector<unsigned int,3> Vector3u;
	typedef Vector<unsigned int,4> Vector4u;
}

// End Definition
#endif // VOX_VECTOR_H