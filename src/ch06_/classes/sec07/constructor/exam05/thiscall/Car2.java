package ch06_.classes.sec07.constructor.exam05.thiscall;
/*
 * 생성자 오버로딩(constructor overloading)
 * 생성자의 이름(class이름)을 중복해서 정의 
 * 조건: 매개변수의 [타입, 갯수, 순서]를 통해서 다르게 정의(다르다는 것을 식별)
 */
public class Car2 {
	// 필드 선언
	String company = "현대자동차";
	String model;
	String color;
	int maxSpeed;
	//같은 클래스의 다른 생성자를 호출할 때 사용됩니다. 이를 '생성자 체이닝(Constructor Chaining)'이라고 합니다.
	Car2(String model) {
		//20라인 생성자 호출
		this.model = model; // 다른 생성자 호출
	}
	
	

	Car2(String model, String color) {
		this(model); // 다른 생성자 호출
		this.color = color;
		// this(model); 
		//: 위와 같이 뒤에 생성해주면 오류(가장 우선적으로 실행필요)
		// 다른 생성자를 호출하는 명령이 가장 앞에 기술되어야함.
		// Constructor call must be the first statement in a constructor
	}

	Car2(String model, String color, int maxSpeed) {
		this(model, color); // 묶어서 객체생성
		this.maxSpeed = maxSpeed;
		
	}
}
	