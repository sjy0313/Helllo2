package ch02_.variable_type.sec12_Printf;

public class PrintfExample {
	public static void main(String[] args) {
		int value = 123;
		System.out.printf("상품의 가격:%d원\n", value); // out f3 -> printstream 객체변수
		System.out.printf("상품의 가격:%6d원\n", value); // 6자리 정수 왼쪽 빈자리 공백
		System.out.printf("상품의 가격:%-6d원\n", value); // - : 왼쪽으로 붙힘.
		System.out.printf("상품의 가격:%06d원\n", value); // 왼쪽 빈자리 0으로 채움

		double area = 3.14159 * 10 * 10; 
		System.out.printf("반지름이 %d인 원의 넓이:%10.2f\n", 10, area);
		System.out.printf("12345678901234567890(%%10.2f)\n"); // 전체 10자리중 소숫점도 자리수에 포함 
		System.out.printf("%10.2f(%%10.2f)\n", area); 
		
		String name = "홍길동"; // 한글은 논리적으로 한 글자이지만 차지하는 공간은 2자리
		String job = "도적";
		System.out.printf("[12345678901234567890123456789012345]\n"); 
		System.out.printf("[%6d|%-10s|%10s]\n", 1, name, job); // 3글자(홍길동) 공간 6자리 차지
		
		String namex = "Andrew";
		System.out.printf("[12345678901234567890123456789012345]\n");
		System.out.printf("[%-10s]\n", namex); // 반면에 영어 6글자(Andrew) 공간 6자리 차지
		
		
	}
}	