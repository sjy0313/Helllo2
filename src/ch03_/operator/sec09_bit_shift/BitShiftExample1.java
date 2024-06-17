/*
 * 비트 이동 연산자 : 
 * << :왼쪽으로 이동, 오른쪽 빈자리는 0으로 채움
 * >> : 오른쪽으로 이동, 왼쪽 빈자리는 최상위 부호 비트와 같은 값으로 채움
 * >>> : 오른쪽으로 이동, 왼쪽 빈자리는 0으로 채움. 
 */
package ch03_.operator.sec09_bit_shift;
// 진수 변환 : https://m.blog.naver.com/shine_a_light/223381027256
public class BitShiftExample1 {
	public static void main(String[] args) {
		int num1 = 1; // 0000 0001 -> 세자리 이동 0000 1000 [8bit]
		int result1 = num1 << 3;
		// public static double pow(double a, double b) [pow f3]
		int result2 = num1 * (int) Math.pow(2, 3); // 2의 3승 
		// 00
		System.out.println("result1: " + result1); // 8 0000 1000 / -8 1111 1000 ( 1111 0111 + 1 = 1111 1000)
		System.out.println("result2: " + result2); // 8
		
		int num2 = -8; // 1111 1000
		int result3 = num2 >> 3; // 1111 1111 (최상위 부호 1을 0에 채워줌.)
		int result4 = num2 / (int) Math.pow(2, 3);
		System.out.println("result3: " + result3); // -1     
		System.out.println("result4: " + result4); // -1		
	}
}	