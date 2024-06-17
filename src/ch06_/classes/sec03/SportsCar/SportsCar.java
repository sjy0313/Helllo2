package ch06_.classes.sec03.SportsCar; // package -> 마을 / class -> 마을지리
// class 는 객체지향 프로그래밍의 기본단위
// 변수들을 같은 그룹끼리 묶어줌
// 클래스에서 변수 -> 속성(특징) / 함수 -> 메소드(method) 행위(동작)
// 변수들의 값의 변화는 함수에서 통제
// class 청사진(blueprint) 실체를 정의한 (틀) -> 실체를 만들어냄(이것이 바로 객체) object
// 메서드/속성 -> 멤버[함수][field(항목)=data]  
// 자바는 독립적인 함수는 없음 
// 객체 지향에서는 객체간 상호작용 하며 동작한다 이떄 수단은 메소드이며 객체가 다른 객체의 기능을 이용할 때 메소드 호출

// 클래스로 부터 생성된 객체를  클래스의 인스턴스(instance)라고 부름
public class SportsCar {
}
// 빈 class (파이썬에서는 빈 클래스 선언x)
class Tire {
}

/*
public class 클래스이름 {
// 필드(멤버 변수)
데이터타입 변수이름;

// 생성자
public 클래스이름() {
    // 초기화 코드
}

// 메소드
public 반환타입 메소드이름(매개변수) { // 매개변수 = 매개값 -> 메소드가 실행할 떄 필요한 값 
    // 메소드의 동작 정의
    return 반환값;
}

}
public class Person {
    // 필드
    String name;
    int age;

    // 생성자 -> 객체가 생성될 때 호출 
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }

    // 메소드(특징) -> 객체의 동작을 정의 
    public void displayInfo() {
        System.out.println("Name: " + name);
        System.out.println("Age: " + age);
    }
    
    public static void main(String[] args) {
        // 객체 생성
        Person person = new Person("John", 25);
        
        // 메소드 호출
        person.displayInfo();
    }
    main 메소드에서 Person 객체를 생성하고 displayInfo 메소드를 호출하여 객체의 정보를 출력합니다.
}*/