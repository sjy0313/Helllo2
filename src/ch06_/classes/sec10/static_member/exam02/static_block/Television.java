package ch06_.classes.sec10.static_member.exam02.static_block;
/*
 * static block : 
 * - 선언과 동시에 초깃값 지정
 * - 클래스가 메모리에 로딩 될 떄 자동으로 실행
 * - 복잡한 로직을 처리해서 값을 지정할 떄 유용
 */
public class Television {
	static String company = "MyCompany"; // 정적변수
	static String model = "LCD";
	static String info = company + "-" + model;
	// 위와 같이 static block 을 따로 만들어주지 않고도
	// 
	//static String info;
/*
	static { // static block
		info = company + "-" + model;
	}
	*/
	
}