package ch06_.classes.sec07.constructor.exam01.car;

public class Car {
	// java의 class 내부에 메소드와 main메소드 생성 가능 
	//생성자(class name과 동일) 선언  : constructor(생성자) 는 new 연산자로 객체를 생성할 떄 객체의 초기화 역할을 담당 선언 형태는 메소드와 비슷
	// 파이썬 생성자 def =__init__  / 소멸자 def = __del__
	
	// 기본생성자 
	// class 의 생성자 정의 되어 있으면 기본 생성자는 자동으로 만들어지지 않는다
	// 그러므로 다른 생성자를 정의하면 기본생성자를 수동으로 만들어야 함.
	/*
	Car() { // overloading(중복해서 생성자 지정 가능 하지만 파이썬에서 생성불가)
		System.out.println("Car: 기본생성자\n");
		
	}
	*/
	
	Car(String model, String color, int maxSpeed) { 
		System.out.printf("Car: %s,%s,%d\n", model, color, maxSpeed);
	}
	
	
}



