/*
 * 증감연산자
 * 증가 : ++ 1씩 증가 
 * 감소 : -- 1씩 감소
 * + 부호유지 / - 부호변경
 * z = ++x / (z + 1 = z) 
 * x++ = z / ( x + 1 = z) 
 */

package ch03_.operator.sec01_increase_decrease;

public class IncreaseDecreaseOperatorExample {
	public static void main(String[] args) {
		int x = 10;
		int y = 10;
		int z; // 로컬변수를 선언해주고 값을 할당 x
		
		x++; // 11   (python : x = x + 1 / x += 1 그리고 초기값 할당 필수 z=0)
		++x; // 12 
		System.out.println("x=" + x);		//12 
		

		System.out.println("-----------------------");	
		
		y--;   // y= 9 , 10 - 1 (증감연산자가 피연산자 뒤에 존재하면 다른 연산 수행 후 실행)
		--y;   // y= 8, 
		System.out.println("y=" + y);		// 8

		System.out.println("-----------------------");		
		z = x++;  // x변수에 값을 z에 할당 후에 x의 값을 1증가 
		System.out.println("z=" + z); // z = 12
		System.out.println("x=" + x);		// ++가 x 뒤에 있으므로
		// z = 12 / x = 13 
		
		System.out.println("-----------------------");		
		z = ++x;   // 14  , x의 값을 1 증가시킨 후 증가된 값을 z에 할당
		System.out.println("z=" + z);   // z = 14
		System.out.println("x=" + x);   // x = 14 
		
		System.out.println("-----------------------");				
		z = ++x + y++; // x의 값을 1 증가 시킨 후 y의 값과 더한 후에 할당하고
		// y의 값을 1증가 시킨다.
		System.out.println("z=" + z);	// 23
		System.out.println("x=" + x);	// 15
		System.out.println("y=" + y);	// 9
	}
}