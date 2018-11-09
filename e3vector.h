/*
 * E3Vector.h - implements the class E3Vector that's intended
 *              to be an STL-compliant math vector in R^3.
 * Optimized 2018 by Joerg Dentler
 *
 */

#ifndef E3_VECTOR_H
#define E3_VECTOR_H

#include <immintrin.h>
#include <tuple>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#if !defined ( __AVX2__ )
#error "no AVX2 configuration"
#endif


class E3Vector
{
public:
    E3Vector()
    {  }
    E3Vector(const double &v)
    {
        fill(v);
    }
    E3Vector(const double v[3])
    {
        m_v = _mm256_setr_pd(v[0], v[1], v[2], 0.0);
    }
    E3Vector(double x, double y, double z)
    {
        m_v = _mm256_setr_pd(x, y, z, 0.0);
    }
    E3Vector(const E3Vector &c)
    {
        m_v = c.m_v;
    }
    E3Vector(const __m256d &v)
    {
        m_v = v;
    }

    /*!
       * \fn size_t size() const
       * \brief Returns the size of this vector.

       The size of this vector is fixed  - \e 3.
    */
    size_t size() const
    {
        return 3;
    }

    /*!
       * \fn void fill(const_reference value)
       * \brief Fills the vector with \e value
       * \param value The value to fill the vector with.

       All the \e n elements of the vector will now be equal
       to \e value.
    */
    void fill(const double &value)
    {
        m_v = _mm256_setr_pd(value, value, value, 0.0);
    }

    /*!
       * \brief Operator [].
       * \param i The index of an element. [0..2]
       * \return the \e i-th element.
       It isn't possible to change the contents of this vector, since
       the data resides in the AVX registers.
    */
    double operator[](size_t i) const
    {
        assert(i < 3);
        double* res = (double*)&m_v;
        return res[i];
    }


    /*!
       * \fn E3Vector operator+(const E3Vector& v) const
       * \brief Addition of two vectors.
       * \param v The vector to be added.
       * \return A vector which is the result of adding \e v to this vector.

       Adds two vectors. The \e i-th element of the result is the sum
       of the \e i-th element of this vector and the \e i-th elements of \e v.
    */
    E3Vector operator+(const E3Vector& v) const
    {
        return _mm256_add_pd(m_v, v.m_v);
    }

    /*!
       * \fn E3Vector operator-(const E3Vector& v) const
       * \brief Subtraction of a vector from a vector.
       * \param v The vector to be subtracted.
       * \return A vector which is the result of subtracting \e v from this vector.

       Subtracts \e v from this vector. The \e i-th element of the result is the
       subtraction of the \e i-th element of \e from the \e i-th element of this vector.
    */
    E3Vector operator-(const E3Vector& v) const
    {

        return _mm256_sub_pd(m_v, v.m_v);
    }

    /*!
       * \fn E3Vector operator*(const_reference scalar) const
       * \brief Multiplication of a vector by a scalar.
       * \param scalar The scalar to be multiplied.
       * \return A vector which is the result of multiplying this vector by \e scalar.

       Multiplies this vector by a scalar. The \e i-th element of the result is the
       multiplication of the \e i-th element of this vector by \e scalar.
    */
    E3Vector operator*(const double &scalar) const
    {
        __m256d v = _mm256_setr_pd(scalar, scalar, scalar, 0.0);
        return _mm256_mul_pd(m_v, v);
    }

    /*!
       * \fn E3Vector operator/(const_reference scalar) const
       * \brief Division of a vector by a scalar.
       * \param scalar The scalar to divide the vector by.
       * \return A vector which is the result of dividing this vector by \e scalar.

       Divides this vector by a scalar. The \e i-th element of the result is the
       division of the \e i-th element of this vector by \e scalar.
    */
    E3Vector operator/(const double &scalar) const
    {
        __m256d v = _mm256_setr_pd(scalar, scalar, scalar, 1.0);
        return _mm256_div_pd(m_v, v);
    }

    /*!
       * \fn E3Vector operator*(const E3Vector& v) const
       * \brief Dot product.
       * \param v A vector.
       * \return The scalar product of this vector and \e v.

       The scalar product of two vectors is the summation of the \e i-th element of
       the first multiplied by the \e i-th element of the other.
    */
    double operator*(const E3Vector& v) const
    {
        __m256d p = _mm256_mul_pd(m_v, v.m_v);
        __m256d s = _mm256_hadd_pd(p, p);
        p = _mm256_permute2f128_pd (s, s, 0b0101);
        p = _mm256_shuffle_pd(s, p, 0b0100);
        s = _mm256_hadd_pd(p, p);
        double* res = (double*)&s;
        return *res;
    }

    /*!
       * \fn mag
       * \brief The magnitude of the vector.
       * \return The magnitude of the vector.

       The scalar product of two vectors is the summation of the \e i-th element of
       the first multiplied by the \e i-th element of the other.
    */
    double mag() const
    {
        return *this * *this;
    }

    /*!
       * \fn E3Vector& operator+=(const E3Vector& v)
       * \brief Addition of two vectors.
       * \param v The vector to be added to this vector.
       * \return A reference to this vector.

       Adds \e v to this vector.
    */
    E3Vector& operator+=(const E3Vector& v)
    {
        m_v = _mm256_add_pd(m_v, v.m_v);
        return *this;
    }

    /*!
       * \fn E3Vector& operator-=(const E3Vector& v)
       * \brief Subtraction of a vector from a vector.
       * \param v The vector to be subtracted from this vector.
       * \return A reference to this vector.

       Subtracts \e v from this vector.
    */
    E3Vector& operator-=(const E3Vector& v)
    {
        m_v = _mm256_sub_pd(m_v, v.m_v);
        return *this;
    }

    /*!
       * \fn E3Vector& operator*=(const_reference scalar)
       * \brief Multiplication of a vector by a scalar.
       * \param scalar The scalar this vector will be multiplied by.
       * \return A reference to this vector.

       Multiplies this vector by \e scalar.
    */
    E3Vector& operator*=(const double &scalar)
    {
        __m256d v = _mm256_setr_pd(scalar, scalar, scalar, 0.0);
        m_v = _mm256_mul_pd(m_v, v);
        return *this;
    }

    /*!
       * \fn E3Vector& operator/=(const_reference scalar)
       * \brief Division of a vector by a scalar.
       * \param scalar The scalar this vector will be divided by.
       * \return A reference to this vector.

       Divides this vector by \e scalar.
    */
    E3Vector& operator/=(const double &scalar)
    {
        __m256d v = _mm256_setr_pd(scalar, scalar, scalar, 1.0);
        m_v = _mm256_div_pd(m_v, v);
        return *this;
    }

    /*!
       * \fn bool operator==(const E3Vector& v) const
       * \brief Checks if this vector is equal to \e v.
       * \param v A vector to be compared to this vector.
       * \sa bool operator!=(const E3Vector& v) const

       If the \e i-th elements of this vector equals the \e i-th elements of
       v the vectors are considered equal.
       Take notice, that if both are invalid they're \b not necessarilly equal.
    */
    bool operator==(const E3Vector& v) const
    {
        double* v1 = (double*)&m_v;
        double* v2 = (double*)&v.m_v;
        return v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2];
    }

    /*!
       * \fn bool operator!=(const E3Vector& v) const
       * \brief Checks if this vector isn't equal to \e v.
       * \param v A vector to be compared to this vector.
       * \sa bool operator==(const E3Vector& v) const

       This is equal to !operator==(v).
    */
    bool operator!=(const E3Vector& v) const
    {
        return !(*this == v);
    }

    /*!
       * \fn bool operator<(const E3Vector& v) const
       * \brief Lexicographical compare.
       * \param v A vector to be compared to this vector.
       * \sa bool operator<=(const E3Vector& v) const, operator>(const E3Vector& v) const,
             bool operator>=(const E3Vector& v) const

       If the \e i-th elements of this vector is less then \e i-th elements of
       v it's considered that this vector is less than \e v.
    */
    bool operator<(const E3Vector& v) const
    {
        double* v1 = (double*)&m_v;
        double* v2 = (double*)&v.m_v;
        return std::make_tuple(v1[0], v1[1], v1[2]) < std::make_tuple(v2[0], v2[1], v2[2]);
    }

    /*!
       * \fn double length() const
       * \brief Returns the length of this vector.
       * \sa double norm() const
    */
    double length() const
    {
        return sqrt(mag());
    }

    double abs() const
    {
        return sqrt(mag());
    }

    /*!
       * \fn double norm() const
       * \brief Returns the length of this vector.
       * \sa double length() const

       Same as length().
    */
    double norm() const
    {
        return length();
    }

    /*!
       * \fn E3Vector normalized() const
       * \brief Returns the normalized vector.
       * \sa E3Vector& normalize()

       Normalization is performed by dividing the vector by its length.
       Therefore, one should make sure this vector is not of length 0.
    */
    E3Vector normalized() const
    {
        return (*this/length());
    }

    /*!
       * \fn E3Vector& normalize()
       * \brief Normalize this vector.
       * \return A reference to this vector.
       * \sa E3Vector normalized() const

       Performs normalization on this vector.
    */
    E3Vector& normalize()
    {
        *this /= length();
        return *this;
    }

    /*!
       * \fn E3Vector& neg()
       * \brief Turn vector to opposite direction
       * \return A reference to this vector.
       * \sa E3Vector negated() const

       Performs negation on this vector.
    */
    E3Vector& neg()
    {
        __m256d v = _mm256_setr_pd(-1.0, -1.0, -1.0, 0.0);
        m_v = _mm256_mul_pd(m_v, v);
        return *this;
    }

    /*!
       * \fn bool isOrthogonal(const E3Vector& v, const T& eps) const
       * \brief Checks whether this vector and \e v are orthogonal.
       * \param v The vector whose orthogonality to this vector we want to check.
       * \param eps A boundary value for the orthogonality check.
       * \return \b True if orthogonal, or \b false otherwise.

       Two vectors are considered orthogonal if their scalar product is 0.
       Because of numerical errors it's very unlikely to get exactly 0,
       especially for real-valued types. Thus, we provide the arguments \e eps,
       which serves as a boundary value. If the scalar product is less or equal to
       \e eps the vectors are considered orthogonal, otherwise - they're not.
    */
    bool isOrthogonal(const E3Vector& v,
                      const double& eps = std::numeric_limits<double>::epsilon()) const
    {
        return (std::abs((*this)*v) <= eps);
    }

    /*!
       * \brief cross product
    */
    E3Vector X(const E3Vector& v) const
    {
        __m256d ap1 = _mm256_permute4x64_pd(m_v, 0b00001001);
        __m256d bp1 = _mm256_permute4x64_pd(v.m_v, 0b00010010);

        __m256d ap2 = _mm256_permute4x64_pd(m_v, 0b00010010);
        __m256d bp2 = _mm256_permute4x64_pd(v.m_v, 0b00001001);

        return _mm256_sub_pd(_mm256_mul_pd(ap1, bp1), _mm256_mul_pd(ap2, bp2));
    }

    /*!
       * \fn void write(std::ostream& os) const
       * \brief Serialization to an ostream.
       * \param os The ostream to write to.
       * \sa void read(std::istream& is)

       Writes the elements of this vector to \e os. Each elements is
       separated by a space.
    */
    void write(std::ostream& os) const
    {
        for (size_t i = 0; i < 3 && os.good(); ++i)
        {
            os << m_v[i];
            if (i != 3-1) os << ',';
        }
    }

protected:
    __m256d m_v;
};

inline
E3Vector operator*(const double &scalar, const E3Vector &v)
{
    return v*scalar;
}

inline
double abs(const E3Vector &v)
{
    return v.length();
}

#endif // E3_VECTOR_H

